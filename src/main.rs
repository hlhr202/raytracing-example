use std::{
    io::Write,
    ops::{Add, Mul, Rem, Sub},
};

#[derive(Debug, Clone, Copy)]
struct Vector3D {
    x: f64,
    y: f64,
    z: f64,
}

impl Add for Vector3D {
    type Output = Vector3D;

    fn add(self, other: Vector3D) -> Vector3D {
        Vector3D {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }
}

impl Sub for Vector3D {
    type Output = Vector3D;

    fn sub(self, other: Vector3D) -> Vector3D {
        Vector3D {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }
}

impl Mul<f64> for Vector3D {
    type Output = Vector3D;

    fn mul(self, scalar: f64) -> Vector3D {
        Vector3D {
            x: self.x * scalar,
            y: self.y * scalar,
            z: self.z * scalar,
        }
    }
}

impl Rem for Vector3D {
    type Output = Vector3D;

    fn rem(self, other: Vector3D) -> Vector3D {
        Vector3D {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }
}

impl Vector3D {
    pub fn new(x: f64, y: f64, z: f64) -> Vector3D {
        Vector3D { x, y, z }
    }

    pub fn mult(&self, other: Vector3D) -> Vector3D {
        Vector3D {
            x: self.x * other.x,
            y: self.y * other.y,
            z: self.z * other.z,
        }
    }

    pub fn norm(&mut self) -> Vector3D {
        let len = (self.x * self.x + self.y * self.y + self.z * self.z).sqrt();
        self.x /= len;
        self.y /= len;
        self.z /= len;
        *self
    }

    pub fn dot(&self, other: Vector3D) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }
}

#[derive(Debug, Clone, Copy)]
struct Ray {
    o: Vector3D, // origin
    d: Vector3D, // direction
}

impl Ray {
    pub fn new(o: Vector3D, d: Vector3D) -> Ray {
        Ray { o, d }
    }
}

#[derive(PartialEq)]
enum ReflType {
    Diff,
    Spec,
    Refr,
}

struct Sphere {
    rad: f64,       // radius
    p: Vector3D,    // position
    e: Vector3D,    // emission
    c: Vector3D,    // color
    refl: ReflType, // reflection type
}

impl Sphere {
    pub fn new(rad: f64, p: Vector3D, e: Vector3D, c: Vector3D, refl: ReflType) -> Sphere {
        Sphere { rad, p, e, c, refl }
    }

    pub fn intersect(&self, r: &Ray) -> f64 {
        let op = self.p - r.o;
        let mut t;
        let eps = 1e-4;
        let b = op.dot(r.d);
        let mut det = b * b - op.dot(op) + self.rad * self.rad;
        if det < 0.0 {
            return 0.0;
        } else {
            det = det.sqrt();
        }

        t = b - det;
        if t > eps {
            t
        } else {
            t = b + det;
            if t > eps {
                t
            } else {
                0.0
            }
        }
    }
}

fn clamp(x: f64) -> f64 {
    x.clamp(0.0, 1.0)
}

fn to_int(x: f64) -> u8 {
    (clamp(x).powf(1.0 / 2.2) * 255.0 + 0.5) as u8
}

fn intersect(r: Ray, t: &mut f64, id: &mut usize, spheres: &[Sphere]) -> bool {
    let inf = 1e20;
    *t = inf;

    for (i, sphere) in spheres.iter().enumerate() {
        let d = sphere.intersect(&r);
        if d != 0.0 && d < *t {
            *t = d;
            *id = i;
        }
    }

    *t < 1e20
}

fn radiance(r: Ray, depth: i32, xi: *mut u16, spheres: &[Sphere]) -> Vector3D {
    let depth = depth + 1;
    let mut t: f64 = 0.0;
    let mut id: usize = 0;
    if !intersect(r, &mut t, &mut id, spheres) {
        return Vector3D::new(0.0, 0.0, 0.0);
    }
    let obj = &spheres[id]; // the hit object
    let x = r.o + r.d * t;
    let n = (x - obj.p).norm();
    let nl = if n.dot(r.d) < 0.0 { n } else { n * -1.0 };
    let mut f = obj.c;
    let p = f.x.max(f.y.max(f.z)); // max refl
    if depth > 5 {
        if unsafe { libc::erand48(xi) } < p {
            f = f * (1.0 / p);
        } else {
            return obj.e;
        }
    }
    if obj.refl == ReflType::Diff {
        let r1 = 2.0 * std::f64::consts::PI * unsafe { libc::erand48(xi) };
        let r2 = unsafe { libc::erand48(xi) };
        let r2s = r2.sqrt();
        let w = nl;
        let u = (if w.x.abs() > 0.1 {
            Vector3D::new(0.0, 1.0, 0.0)
        } else {
            Vector3D::new(1.0, 0.0, 0.0)
        } % w)
            .norm();
        let v = w % u;
        let d = (u * (r1.cos() * r2s) + v * (r1.sin() * r2s) + w * (1.0 - r2).sqrt()).norm();
        return obj.e + f.mult(radiance(Ray::new(x, d), depth, xi, spheres));
    } else if obj.refl == ReflType::Spec {
        return obj.e
            + f.mult(radiance(
                Ray::new(x, r.d - n * 2.0 * n.dot(r.d)),
                depth,
                xi,
                spheres,
            ));
    }

    let refl_ray = Ray::new(x, r.d - n * 2.0 * n.dot(r.d));
    let into = n.dot(nl) > 0.0;
    let nc = 1.0;
    let nt = 1.5;
    let nnt = if into { nc / nt } else { nt / nc };
    let ddn = r.d.dot(nl);
    let cos2t = 1.0 - nnt * nnt * (1.0 - ddn * ddn);
    if cos2t < 0.0 {
        return obj.e + f.mult(radiance(refl_ray, depth, xi, spheres));
    }

    let tdir =
        (r.d * nnt - n * ((if into { 1.0 } else { -1.0 }) * (ddn * nnt + cos2t.sqrt()))).norm();
    let a = nt - nc;
    let b = nt + nc;
    let r0 = a * a / (b * b);
    let c = 1.0 - (if into { -ddn } else { tdir.dot(n) });
    let re = r0 + (1.0 - r0) * c.powf(5.0);
    let tr = 1.0 - re;
    let p = 0.25 + 0.5 * re;
    let rp = re / p;
    let tp = tr / (1.0 - p);

    obj.e
        + f.mult(if depth > 2 {
            if unsafe { libc::erand48(xi) } < p {
                radiance(refl_ray, depth, xi, spheres) * rp
            } else {
                radiance(Ray::new(x, tdir), depth, xi, spheres) * tp
            }
        } else {
            radiance(refl_ray, depth, xi, spheres) * re
                + radiance(Ray::new(x, tdir), depth, xi, spheres) * tr
        })
}

fn main() {
    let spheres = [
        Sphere::new(
            1e5,
            Vector3D::new(1e5 + 1.0, 40.8, 81.6),
            Vector3D::new(0.0, 0.0, 0.0),
            Vector3D::new(0.75, 0.25, 0.25),
            ReflType::Diff,
        ), // Left
        Sphere::new(
            1e5,
            Vector3D::new(-1e5 + 99.0, 40.8, 81.6),
            Vector3D::new(0.0, 0.0, 0.0),
            Vector3D::new(0.25, 0.25, 0.75),
            ReflType::Diff,
        ), // Right
        Sphere::new(
            1e5,
            Vector3D::new(50.0, 40.8, 1e5),
            Vector3D::new(0.0, 0.0, 0.0),
            Vector3D::new(0.75, 0.75, 0.75),
            ReflType::Diff,
        ), // Back
        Sphere::new(
            1e5,
            Vector3D::new(50.0, 40.8, -1e5 + 170.0),
            Vector3D::new(0.0, 0.0, 0.0),
            Vector3D::new(0.0, 0.0, 0.0),
            ReflType::Diff,
        ), // Front
        Sphere::new(
            1e5,
            Vector3D::new(50.0, 1e5, 81.6),
            Vector3D::new(0.0, 0.0, 0.0),
            Vector3D::new(0.75, 0.75, 0.75),
            ReflType::Diff,
        ), // Ceiling
        Sphere::new(
            1e5,
            Vector3D::new(50.0, -1e5 + 81.6, 81.6),
            Vector3D::new(0.0, 0.0, 0.0),
            Vector3D::new(0.75, 0.75, 0.75),
            ReflType::Diff,
        ), // Floor
        Sphere::new(
            16.5,
            Vector3D::new(27.0, 16.5, 47.0),
            Vector3D::new(0.0, 0.0, 0.0),
            Vector3D::new(0.999, 0.999, 0.999),
            ReflType::Spec,
        ), // Mirror
        Sphere::new(
            16.5,
            Vector3D::new(73.0, 16.5, 78.0),
            Vector3D::new(0.0, 0.0, 0.0),
            Vector3D::new(0.999, 0.999, 0.999),
            ReflType::Refr,
        ), // Glass
        Sphere::new(
            600.0,
            Vector3D::new(50.0, 681.6 - 0.27, 81.6),
            Vector3D::new(12.0, 12.0, 12.0),
            Vector3D::new(0.0, 0.0, 0.0),
            ReflType::Diff,
        ), // Light
    ];

    let w = 1024;
    let h = 768;
    let args = std::env::args().collect::<Vec<String>>();
    let n_samples = if args.len() > 1 {
        args[1].parse::<i32>().unwrap() / 4
    } else {
        1
    };

    let cam = Ray::new(
        Vector3D::new(50.0, 52.0, 295.6),
        Vector3D::new(0.0, -0.042612, -1.0).norm(),
    );

    let cx = Vector3D::new(w as f64 * 0.5135 / h as f64, 0.0, 0.0);
    let cy = (cx % cam.d).norm() * 0.5135;
    let mut r;
    let mut c = vec![Vector3D::new(0.0, 0.0, 0.0); w * h];

    println!("sample: {}", n_samples * 4);

    for y in 0..h {
        // Loop over image height
        eprintln!(
            "Rendering ({} spp) {}%",
            n_samples * 4,
            // to fixed x.xx
            (100.0 * y as f64 / (h as f64 - 1.0) * 100.0).round() / 100.0
        );
        for x in 0..w {
            // Loop over image width
            let mut xis = [0, 0, (y * y * y) as u16];
            for sy in 0..2 {
                let i = (h - y - 1) * w + x;
                for sx in 0..2 {
                    r = Vector3D::new(0.0, 0.0, 0.0);
                    for _ in 0..n_samples {
                        let r1 = 2.0 * unsafe { libc::erand48(xis.as_mut_ptr()) };
                        let r2 = 2.0 * unsafe { libc::erand48(xis.as_mut_ptr()) };
                        let dx = if r1 < 1.0 {
                            r1.sqrt() - 1.0
                        } else {
                            1.0 - (2.0 - r1).sqrt()
                        };
                        let dy = if r2 < 1.0 {
                            r2.sqrt() - 1.0
                        } else {
                            1.0 - (2.0 - r2).sqrt()
                        };
                        let mut d = cx
                            * (((sx as f64 + 0.5 + dx) / 2.0 + x as f64) / w as f64 - 0.5)
                            + cy * (((sy as f64 + 0.5 + dy) / 2.0 + y as f64) / h as f64 - 0.5)
                            + cam.d;
                        r = r + radiance(
                            Ray::new(cam.o + d * 140.0, d.norm()),
                            0,
                            xis.as_mut_ptr(),
                            &spheres,
                        ) * (1.0 / n_samples as f64);
                    }
                    c[i] = c[i] + Vector3D::new(clamp(r.x), clamp(r.y), clamp(r.z)) * 0.25;
                }
            }
        }
    }

    let file = std::fs::File::create("image.ppm").unwrap();
    let mut writer = std::io::BufWriter::new(file);
    writer
        .write_all(format!("P3\n{} {}\n{}\n", w, h, 255).as_bytes())
        .unwrap();
    for c in c.iter() {
        writer
            .write_all(format!("{} {} {}\n", to_int(c.x), to_int(c.y), to_int(c.z)).as_bytes())
            .unwrap();
    }
}
