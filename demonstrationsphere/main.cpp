// pretty_sphere.cpp
// Single-file CPU ray tracer: glossy sphere, checker plane, soft shadows.
// Writes pretty_sphere.ppm (binary PPM).

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>   // <-- for atoi on MSVC
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

using std::sqrt;
constexpr float INF = std::numeric_limits<float>::infinity();
constexpr float PI  = 3.14159265358979323846f;
inline float deg2rad(float d){ return d * (PI/180.0f); }
inline float clamp(float x, float a, float b){ return std::min(b, std::max(a,x)); }

// ----------------------------- Math -----------------------------
struct Vec3 {
    float x,y,z;
    Vec3():x(0),y(0),z(0) {}
    Vec3(float x_,float y_,float z_):x(x_),y(y_),z(z_) {}
    Vec3 operator-() const { return Vec3(-x,-y,-z); }
    Vec3& operator+=(const Vec3& o){ x+=o.x; y+=o.y; z+=o.z; return *this; }
    Vec3& operator-=(const Vec3& o){ x-=o.x; y-=o.y; z-=o.z; return *this; }
    Vec3& operator*=(float s){ x*=s; y*=s; z*=s; return *this; }
    Vec3& operator/=(float s){ return (*this) *= (1.0f/s); }
    float length() const { return std::sqrt(x*x+y*y+z*z); }
    float length2() const { return x*x+y*y+z*z; }
};
inline Vec3 operator+(const Vec3&a,const Vec3&b){ return Vec3(a.x+b.x,a.y+b.y,a.z+b.z); }
inline Vec3 operator-(const Vec3&a,const Vec3&b){ return Vec3(a.x-b.x,a.y-b.y,a.z-b.z); }
inline Vec3 operator*(const Vec3&a,float s){ return Vec3(a.x*s,a.y*s,a.z*s); }
inline Vec3 operator*(float s,const Vec3&a){ return a*s; }
inline Vec3 operator/(const Vec3&a,float s){ return Vec3(a.x/s,a.y/s,a.z/s); }
// **** component-wise operators (fix for ambient * base, etc.) ****
inline Vec3 operator*(const Vec3& a, const Vec3& b){ return Vec3(a.x*b.x, a.y*b.y, a.z*b.z); }
inline Vec3 operator/(const Vec3& a, const Vec3& b){ return Vec3(a.x/b.x, a.y/b.y, a.z/b.z); }

inline float dot(const Vec3&a,const Vec3&b){ return a.x*b.x+a.y*b.y+a.z*b.z; }
inline Vec3 cross(const Vec3&a,const Vec3&b){
    return Vec3(a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x);
}
inline Vec3 normalize(const Vec3&v){ float L=v.length(); return (L>0)? v*(1.0f/L): v; }
inline Vec3 reflect(const Vec3&v,const Vec3&n){ return v - n*(2.0f*dot(v,n)); }

// ----------------------------- RNG ------------------------------
struct RNG {
    std::mt19937 gen;
    std::uniform_real_distribution<float> dist{0.0f,1.0f};
    RNG(uint64_t seed){ gen.seed((uint32_t)seed); }
    float uniform(){ return dist(gen); }
};

inline Vec3 sampleDisk(RNG& rng, const Vec3& center, const Vec3& ux, const Vec3& uy, float R){
    float r = std::sqrt(rng.uniform()) * R;
    float t = 2.f*PI*rng.uniform();
    return center + ux*(r*std::cos(t)) + uy*(r*std::sin(t));
}

inline Vec3 sampleCone(RNG& rng, const Vec3& dir, float theta){
    Vec3 w = normalize(dir);
    Vec3 a = (std::abs(w.x) > 0.1f) ? Vec3(0,1,0) : Vec3(1,0,0);
    Vec3 u = normalize(cross(a,w));
    Vec3 v = cross(w,u);
    float cosT = 1.0f - rng.uniform()*(1.0f - std::cos(theta));
    float sinT = std::sqrt(1.0f - cosT*cosT);
    float phi  = 2.f*PI*rng.uniform();
    return normalize(u*(std::cos(phi)*sinT) + v*(std::sin(phi)*sinT) + w*cosT);
}

// ----------------------------- Ray ------------------------------
struct Ray {
    Vec3 o, d;
    Ray() {}
    Ray(const Vec3&o_, const Vec3&d_):o(o_),d(normalize(d_)){}
    Vec3 at(float t) const { return o + d*t; }
};

// --------------------------- Camera -----------------------------
struct Camera {
    Vec3 pos, forward, right, up;
    float vfov, aspect;
    Camera(Vec3 lookfrom, Vec3 lookat, Vec3 vup, float vfov_deg, float aspect_)
        : pos(lookfrom), vfov(vfov_deg), aspect(aspect_) {
        forward = normalize(lookat - lookfrom);
        right   = normalize(cross(forward, vup));
        up      = cross(right, forward);
    }
    Ray get_ray(RNG&, float u, float v){
        float h = std::tan(deg2rad(vfov)*0.5f);
        float w = h * aspect;
        Vec3 dir = normalize(forward + right*((2*u-1)*w) + up*((2*v-1)*h));
        return Ray(pos, dir);
    }
};

// --------------------------- Scene ------------------------------
struct Hit {
    float t = INF;
    Vec3 p;
    Vec3 n;
    int  id = -1; // 0 = sphere, 1 = plane
};

struct Sphere {
    Vec3 c; float r;
    Vec3 albedo;   // base color
    float metal;   // 0..1
    float rough;   // 0..1 (for glossy reflection cone)
    float spec;    // specular strength (Blinn-Phong)
    float shininess;

    bool intersect(const Ray& ray, float tMin, float tMax, Hit& rec) const {
        Vec3 oc = ray.o - c;
        float a = dot(ray.d, ray.d);
        float b = dot(oc, ray.d);
        float c2 = dot(oc, oc) - r*r;
        float disc = b*b - a*c2;
        if (disc < 0) return false;
        float s = std::sqrt(disc);
        float t = (-b - s)/a;
        if (t < tMin || t > tMax){
            t = (-b + s)/a;
            if (t < tMin || t > tMax) return false;
        }
        rec.t = t;
        rec.p = ray.at(t);
        rec.n = normalize(rec.p - c);
        rec.id = 0;
        return true;
    }
};

struct Plane {
    // y = 0 plane, checker pattern
    Vec3 albedo1{0.80f, 0.80f, 0.80f};
    Vec3 albedo2{0.15f, 0.15f, 0.15f};
    float spec = 0.05f;
    float shininess = 32.0f;

    bool intersect(const Ray& ray, float tMin, float tMax, Hit& rec) const {
        if (std::abs(ray.d.y) < 1e-5f) return false;
        float t = -ray.o.y / ray.d.y;
        if (t < tMin || t > tMax) return false;
        rec.t = t;
        rec.p = ray.at(t);
        rec.n = Vec3(0,1,0);
        rec.id = 1;
        return true;
    }
    Vec3 colorAt(const Vec3& p) const {
        int ix = (int)std::floor(p.x);
        int iz = (int)std::floor(p.z);
        bool even = ((ix + iz) & 1) == 0;
        return even ? albedo1 : albedo2;
    }
};

// -------------------------- Lighting ----------------------------
struct AreaLight {
    Vec3 center;
    Vec3 u, v;     // orthonormal basis spanning the light disk
    float radius;
    Vec3 intensity; // radiance color (acts like color * power)

    Vec3 samplePoint(RNG& rng) const {
        return sampleDisk(rng, center, u, v, radius);
    }
};

// ------------------------ Scene Shading -------------------------
struct Scene {
    Sphere sphere;
    Plane  plane;
    AreaLight light;
    Vec3  ambient{0.03f,0.03f,0.03f};
    Vec3  skyTop{0.70f,0.85f,1.00f};
    Vec3  skyBot{1.00f,1.00f,1.00f};

    bool hit(const Ray& r, float tMin, float tMax, Hit& h) const {
        Hit temp; bool hitAnything = false;
        if (sphere.intersect(r, tMin, tMax, temp)){ h = temp; tMax = temp.t; hitAnything = true; }
        if (plane.intersect(r, tMin, tMax, temp)){ h = temp; hitAnything = true; }
        return hitAnything;
    }

    bool occluded(const Ray& r, float tMin, float tMax) const {
        Hit h;
        return hit(r, tMin, tMax, h);
    }

    Vec3 sky(const Ray& r) const {
        float t = 0.5f*(r.d.y + 1.0f);
        return skyBot*(1.0f - t) + skyTop*t;
    }

    static Vec3 fresnelSchlick(float cosTheta, const Vec3& F0){
        return F0 + (Vec3{1,1,1} - F0) * std::pow(1.0f - cosTheta, 5.0f);
    }

    Vec3 shade(const Ray& r, RNG& rng, int depth) const {
        if (depth <= 0) return Vec3(0,0,0);

        Hit h;
        if (!hit(r, 0.001f, INF, h)) {
            return sky(r);
        }

        Vec3 p = h.p;
        Vec3 n = h.n;

        // Sample area light once per primary sample (soft shadow accumulates with AA)
        Vec3 Lpos = light.samplePoint(rng);
        Vec3 Ldir = Lpos - p;
        float distL = Ldir.length();
        Ldir = Ldir / distL;

        // Shadow ray
        bool shadowed = occluded(Ray(p + n*0.001f, Ldir), 0.001f, distL - 0.001f);

        // Light falloff
        float atten = 1.0f / (distL*distL);
        Vec3 Lo(0,0,0);

        // Common terms
        float NdotL = clamp(dot(n, Ldir), 0.f, 1.f);
        Vec3 H = normalize(Ldir - r.d); // half-vector for Blinn-Phong
        float NdotH = clamp(dot(n, H), 0.f, 1.f);

        if (h.id == 0) {
            // Sphere shading: glossy/metallic with reflection
            const Sphere& sp = sphere;
            Vec3 base = sp.albedo;

            if (!shadowed) {
                Vec3 diffuse = base * NdotL;
                float specTerm = std::pow(NdotH, sp.shininess) * sp.spec;
                Vec3 specular = Vec3(specTerm, specTerm, specTerm);
                Lo += (diffuse * (1.0f - sp.metal) + specular) * light.intensity * atten;
            }

            // Reflection (one bounce), glossy controlled by roughness
            Vec3 F0 = sp.metal > 0.5f ? base : Vec3(0.04f, 0.04f, 0.04f);
            float cosTheta = clamp(dot(-r.d, n), 0.f, 1.f);
            Vec3 F = fresnelSchlick(cosTheta, F0);

            Vec3 reflDirIdeal = reflect(r.d, n);
            float maxCone = deg2rad(20.0f);
            float cone = sp.rough * maxCone;
            Vec3 reflDir = cone > 1e-4f ? sampleCone(rng, reflDirIdeal, cone) : reflDirIdeal;

            Vec3 refl = shade(Ray(p + n*0.002f, reflDir), rng, depth-1);
            Vec3 kS = F;                                          // reflection weight
            Vec3 kD = (Vec3{1,1,1} - kS) * (1.0f - sp.metal);     // leftover diffuse
            return ambient*base + kD*Lo + kS*refl;                // <-- uses Vec3*Vec3
        } else {
            // Plane shading (checker lambert + small specular)
            Vec3 base = plane.colorAt(p);
            Vec3 color = ambient * base;                          // <-- uses Vec3*Vec3

            if (!shadowed) {
                Vec3 diffuse = base * NdotL;
                float specTerm = std::pow(NdotH, plane.shininess) * plane.spec;
                Vec3 specular = Vec3(specTerm, specTerm, specTerm);
                color += (diffuse + specular) * light.intensity * atten; // Vec3*Vec3
            }
            return color;
        }
    }
};

// --------------------------- Rendering --------------------------
struct RenderParams {
    int W = 1920;
    int H = 1080;
    int SPP = 64;
    int maxDepth = 4;
};

int main(int argc, char** argv){
    RenderParams p;
    if (argc >= 2) p.W = std::max(16, std::atoi(argv[1]));
    if (argc >= 3) p.H = std::max(16, std::atoi(argv[2]));
    if (argc >= 4) p.SPP = std::max(1,  std::atoi(argv[3]));

    float aspect = float(p.W)/float(p.H);

    Camera cam(
        /*lookfrom*/ Vec3(0.0f, 1.0f, 5.0f),
        /*lookat  */ Vec3(0.0f, 1.0f, 0.0f),
        /*vup     */ Vec3(0.0f, 1.0f, 0.0f),
        /*vfov    */ 45.0f,
        aspect
    );

    Scene scene;
    scene.sphere = Sphere{
        /*c*/ Vec3(0.0f, 1.0f, 0.0f),
        /*r*/ 1.0f,
        /*albedo*/ Vec3(0.95f, 0.35f, 0.20f),
        /*metal*/ 0.85f,
        /*rough*/ 0.15f,
        /*spec*/  0.8f,
        /*shininess*/ 128.0f
    };

    Vec3 Lcenter(2.0f, 4.0f, 2.0f);
    Vec3 Lw = normalize(Vec3(-0.4f, -1.0f, -0.35f));
    Vec3 Ltmp = (std::abs(Lw.y) < 0.9f) ? Vec3(0,1,0) : Vec3(1,0,0);
    Vec3 Lu = normalize(cross(Ltmp, Lw));
    Vec3 Lv = cross(Lw, Lu);
    scene.light = AreaLight{
        Lcenter, Lu, Lv,
        /*radius*/ 0.75f,
        /*intensity*/ Vec3(24.0f, 24.0f, 24.0f)
    };

    std::vector<uint8_t> img(p.W * p.H * 3, 0);

    int threads = std::max(1u, std::thread::hardware_concurrency());
    std::atomic<int> nextRow{0};
    std::vector<std::thread> pool;
    pool.reserve(threads);

    auto worker = [&](int tid){
        uint64_t seed = (uint64_t)std::chrono::high_resolution_clock::now().time_since_epoch().count();
        seed ^= (0x9E3779B97F4A7C15ull + (uint64_t)tid + (seed<<6) + (seed>>2));
        RNG rng(seed);

        while (true) {
            int j = nextRow.fetch_add(1);
            if (j >= p.H) break;

            for (int i = 0; i < p.W; ++i) {
                Vec3 col(0,0,0);
                for (int s = 0; s < p.SPP; ++s) {
                    float u = (i + rng.uniform()) / (float)p.W;
                    float v = (j + rng.uniform()) / (float)p.H;
                    Ray r = cam.get_ray(rng, u, 1.0f - v);
                    col += scene.shade(r, rng, p.maxDepth);
                }
                col /= float(p.SPP);
                col = Vec3(std::sqrt(col.x), std::sqrt(col.y), std::sqrt(col.z)); // gamma
                int idx = (j * p.W + i) * 3;
                img[idx+0] = (uint8_t)(clamp(col.x,0.f,1.f) * 255.99f);
                img[idx+1] = (uint8_t)(clamp(col.y,0.f,1.f) * 255.99f);
                img[idx+2] = (uint8_t)(clamp(col.z,0.f,1.f) * 255.99f);
            }
        }
    };

    for (int t = 0; t < threads; ++t) pool.emplace_back(worker, t);
    for (auto& th : pool) th.join();

    std::ofstream ofs("pretty_sphere.ppm", std::ios::binary);
    ofs << "P6\n" << p.W << " " << p.H << "\n255\n";
    ofs.write(reinterpret_cast<const char*>(img.data()), img.size());
    ofs.close();
    return 0;
}
