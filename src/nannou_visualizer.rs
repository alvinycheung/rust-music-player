use nannou::prelude::*;
use std::sync::{Arc, Mutex};
use std::path::PathBuf;
use std::fs;
use std::io::BufReader;
use std::time::{Duration, Instant};
use rand::seq::SliceRandom;
use rand::Rng;
use rustfft::{FftPlanner, num_complex::Complex};
use rodio::{Decoder, OutputStream, Sink, Source};
use std::thread;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use id3::Tag;
use id3::TagLike;
use std::io::Write;

// Constants
const NUM_BANDS: usize = 180 * 3;  // Number of frequency bands (points around circle)
const HALF_BANDS: usize = NUM_BANDS / 2;  // Half the bands for symmetry
const SPIKES: usize = 6;  // Number of spikes around the circle
const SECTIONS: usize = SPIKES * 2;  // Total sections (12) - one upright & one reversed per spike
const SECTION_SIZE: usize = NUM_BANDS / SECTIONS;  // Size of each section for spikes
const CIRCLE_RADIUS: f32 = 100.0;  // Base radius for visualization
const MIN_INNER_RADIUS: f32 = 30.0;  // Minimum inner radius when fully contracted
const MAX_AMPLITUDE: f32 = 200.0;  // Maximum amplitude for visualization
const TRACER_LENGTH: usize = 40;  // History length for tracers
const MUSIC_DIR: &str = "src/music";
const WINDOW_WIDTH: u32 = 1000;
const WINDOW_HEIGHT: u32 = 1000;
const FFT_SIZE: usize = 4096;  // FFT size (power of 2)
const DROP_SPEED: f32 = 15.0;  // Speed at which bands drop toward center when amplitude decreases
const RISE_SPEED: f32 = 6.0;  // Speed at which bands rise back to normal position
const ROTATION_OFFSET: f32 = PI * 0.5;  // Rotate counter-clockwise by 90 degrees (PI/2 radians)
const OUTWARD_SLOWDOWN_FACTOR: f32 = 0.4;  // How much to slow down movement when extended outward
const INWARD_SPEEDUP_FACTOR: f32 = 1.5;  // How much to speed up movement when returning to center
const INNER_RADIUS_DROP_FACTOR: f32 = 0.6;  // How much the inner radius contracts with velocity
const MAX_STARS: usize = 100;  // Maximum number of stars in the star field
const STAR_FIELD_DEPTH: f32 = 1000.0;  // Z-depth of the star field
const STAR_SPEED_BASE: f32 = 100.0;  // Base speed of stars moving toward viewer
const STAR_SPEED_MUSIC_FACTOR: f32 = 300.0;  // How much music affects star speed
const MERKABA_SIZE: f32 = 40.0;  // Base size of the merkaba
const MERKABA_REFLECTION_STRENGTH: f32 = 0.7;  // Strength of reflections (0-1)

// Add constants for volume control and display
const DEFAULT_VOLUME: f32 = 1.0;
const VOLUME_CHANGE_STEP: f32 = 0.05;
const VOLUME_BAR_WIDTH: f32 = 1.0;
const VOLUME_BAR_MARGIN: f32 = 20.0;
const VOLUME_INDICATOR_FADE_TIME: f32 = 3.0;
const SETTINGS_FILE: &str = "visualizer_settings.json";

// 3D vector for merkaba vertices
#[derive(Clone, Copy)]
struct Vec3 {
    x: f32,
    y: f32,
    z: f32,
}

impl Vec3 {
    fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    // Project 3D point to 2D screen space with perspective
    fn project(&self, z_offset: f32) -> Point2 {
        let perspective_factor = 500.0;
        let z_perspective = perspective_factor / (perspective_factor + self.z + z_offset);

        Point2::new(
            self.x * z_perspective,
            self.y * z_perspective
        )
    }
}

// Face for the merkaba
#[derive(Clone)]
struct Face {
    vertices: [Vec3; 3],
    color: Rgba,
    normal: Vec3,
}

// Merkaba (Star Tetrahedron) visualization
struct Merkaba {
    faces: Vec<Face>,
    rotation: [f32; 3],  // [x, y, z] rotation angles
    position: Vec3,      // Center position
    size: f32,           // Current size
    base_size: f32,      // Base size
    reflection_points: Vec<Point2>,  // Points to use for reflection sampling
}

impl Merkaba {
    fn new() -> Self {
        // Create two tetrahedrons (one pointing up, one pointing down)
        let mut faces = Vec::new();
        let base_size = MERKABA_SIZE;

        // Both tetrahedrons should be centered at exactly the same position (origin)
        // Modified vertices to ensure they share the same center (0,0,0)

        // Upward tetrahedron vertices - centered at origin
        let up_vertices = [
            Vec3::new(0.0, -0.8, 0.0),             // Top
            Vec3::new(-0.8, 0.3, 0.0),           // Bottom left
            Vec3::new(0.8, 0.3, 0.0),            // Bottom right
            Vec3::new(0.0, 0.0, 0.8),      // Back
        ];

        // Downward tetrahedron vertices - centered at origin
        let down_vertices = [
            Vec3::new(0.0, 0.8, 0.0),              // Bottom
            Vec3::new(-0.8, -0.3, 0.0),          // Top left
            Vec3::new(0.8, -0.3, 0.0),           // Top right
            Vec3::new(0.0, 0.0, -0.8),     // Front
        ];

        // Helper function to calculate face normal
        let calculate_normal = |v1: &Vec3, v2: &Vec3, v3: &Vec3| -> Vec3 {
            // Cross product of two edges
            let edge1 = Vec3::new(v2.x - v1.x, v2.y - v1.y, v2.z - v1.z);
            let edge2 = Vec3::new(v3.x - v1.x, v3.y - v1.y, v3.z - v1.z);

            // Cross product
            let cross_x = edge1.y * edge2.z - edge1.z * edge2.y;
            let cross_y = edge1.z * edge2.x - edge1.x * edge2.z;
            let cross_z = edge1.x * edge2.y - edge1.y * edge2.x;

            // Normalize
            let length = (cross_x * cross_x + cross_y * cross_y + cross_z * cross_z).sqrt();
            Vec3::new(cross_x / length, cross_y / length, cross_z / length)
        };

        // Create upward tetrahedron faces with gold reflective color
        let up_color = rgba(1.0, 0.84, 0.0, 0.85);  // Golden color

        // Face 1: Top, Bottom left, Bottom right
        faces.push(Face {
            vertices: [up_vertices[0], up_vertices[1], up_vertices[2]],
            color: up_color,
            normal: calculate_normal(&up_vertices[0], &up_vertices[1], &up_vertices[2]),
        });

        // Face 2: Top, Bottom left, Back
        faces.push(Face {
            vertices: [up_vertices[0], up_vertices[1], up_vertices[3]],
            color: up_color,
            normal: calculate_normal(&up_vertices[0], &up_vertices[1], &up_vertices[3]),
        });

        // Face 3: Top, Bottom right, Back
        faces.push(Face {
            vertices: [up_vertices[0], up_vertices[2], up_vertices[3]],
            color: up_color,
            normal: calculate_normal(&up_vertices[0], &up_vertices[2], &up_vertices[3]),
        });

        // Face 4: Bottom left, Bottom right, Back
        faces.push(Face {
            vertices: [up_vertices[1], up_vertices[2], up_vertices[3]],
            color: up_color,
            normal: calculate_normal(&up_vertices[1], &up_vertices[2], &up_vertices[3]),
        });

        // Create downward tetrahedron faces with silver reflective color
        let down_color = rgba(0.75, 0.75, 0.95, 0.85);  // Silver color

        // Face 1: Bottom, Top left, Top right
        faces.push(Face {
            vertices: [down_vertices[0], down_vertices[1], down_vertices[2]],
            color: down_color,
            normal: calculate_normal(&down_vertices[0], &down_vertices[1], &down_vertices[2]),
        });

        // Face 2: Bottom, Top left, Front
        faces.push(Face {
            vertices: [down_vertices[0], down_vertices[1], down_vertices[3]],
            color: down_color,
            normal: calculate_normal(&down_vertices[0], &down_vertices[1], &down_vertices[3]),
        });

        // Face 3: Bottom, Top right, Front
        faces.push(Face {
            vertices: [down_vertices[0], down_vertices[2], down_vertices[3]],
            color: down_color,
            normal: calculate_normal(&down_vertices[0], &down_vertices[2], &down_vertices[3]),
        });

        // Face 4: Top left, Top right, Front
        faces.push(Face {
            vertices: [down_vertices[1], down_vertices[2], down_vertices[3]],
            color: down_color,
            normal: calculate_normal(&down_vertices[1], &down_vertices[2], &down_vertices[3]),
        });

        // Initialize reflection points
        let reflection_points = Vec::new();

        Self {
            faces,
            rotation: [0.0, 0.0, 0.0],
            position: Vec3::new(0.0, 0.0, 0.0),
            size: base_size,
            base_size,
            reflection_points,
        }
    }

    fn update(&mut self, delta_time: f32, audio_data: &AudioData, reflection_points: Vec<Point2>, mouse_pos: Point2) {
        // Update rotation based on audio
        let power = audio_data.power.min(1.0);
        let base_speed_factor = 0.2 + power * 0.3;  // Base rotation speed + audio reactivity

        // Calculate distance from mouse to center
        let mouse_distance = (mouse_pos.x * mouse_pos.x + mouse_pos.y * mouse_pos.y).sqrt();

        // Define the radius of influence (how far the mouse affects the merkaba)
        let influence_radius = 300.0;

        // Calculate mouse influence factor (stronger when mouse is closer)
        let mouse_influence = if mouse_distance < influence_radius {
            let normalized_distance = (influence_radius - mouse_distance) / influence_radius;
            ease_in_out_cubic(normalized_distance) // Apply easing for smooth transition
        } else {
            0.0
        };

        // Calculate mouse angle relative to center
        let mouse_angle = if mouse_distance > 5.0 {
            (mouse_pos.y.atan2(mouse_pos.x) + PI) % (2.0 * PI)
        } else {
            0.0 // Prevent jittering when mouse is very close to center
        };

        // Create rotation vectors from mouse position
        let mouse_y_rot = -mouse_pos.x / influence_radius * PI * 0.25; // Tilt toward/away from mouse X
        let mouse_x_rot = mouse_pos.y / influence_radius * PI * 0.25;  // Tilt toward/away from mouse Y
        let mouse_z_rot = mouse_angle * 0.1;                          // Subtle spin based on angle

        // Different rotation rates for each axis
        // Base rotation with audio reactivity
        let x_speed = delta_time * 0.2 * base_speed_factor;
        let y_speed = delta_time * 0.3 * base_speed_factor;
        let z_speed = delta_time * 0.1 * base_speed_factor;

        // Update rotation with blend between automatic rotation and mouse-influenced rotation
        if mouse_influence > 0.0 {
            // Blend between automatic rotation and mouse-influenced rotation
            self.rotation[0] += x_speed * (1.0 - mouse_influence) +
                                (mouse_x_rot - self.rotation[0]) * mouse_influence * delta_time * 5.0;

            self.rotation[1] += y_speed * (1.0 - mouse_influence) +
                                (mouse_y_rot - self.rotation[1]) * mouse_influence * delta_time * 5.0;

            // For z-axis, add regular rotation plus a mouse-direction component
            self.rotation[2] += z_speed * (1.0 - mouse_influence * 0.5) +
                                mouse_z_rot * mouse_influence * delta_time * 3.0;
        } else {
            // Regular rotation when mouse is far
            self.rotation[0] += x_speed;
            self.rotation[1] += y_speed;
            self.rotation[2] += z_speed;
        }

        // Make rotation values loop between 0 and 2*PI
        for i in 0..3 {
            self.rotation[i] = self.rotation[i] % (2.0 * PI);
        }

        // Update size with audio reactivity - pulsate with the beat
        let bass_power = audio_data.spectrum.iter()
            .take(SECTION_SIZE / 4)
            .sum::<f32>() / (SECTION_SIZE / 4) as f32;

        // Apply elastic easing for more dramatic effect
        let audio_size_factor = 1.0 + ease_out_elastic(bass_power) * 0.3;

        // Add a subtle size increase when mouse is close
        let mouse_size_factor = 1.0 + mouse_influence * 0.15;

        self.size = self.base_size * audio_size_factor * mouse_size_factor;

        // Store reflection points
        self.reflection_points = reflection_points;
    }

    fn draw(&self, draw: &Draw, audio_data: &AudioData) {
        let sin_x = self.rotation[0].sin();
        let cos_x = self.rotation[0].cos();
        let sin_y = self.rotation[1].sin();
        let cos_y = self.rotation[1].cos();
        let sin_z = self.rotation[2].sin();
        let cos_z = self.rotation[2].cos();

        // Sort faces back to front for proper depth rendering
        let mut sorted_faces: Vec<(Face, f32)> = self.faces.iter().map(|face| {
            // Find center point of the face
            let center = Vec3::new(
                (face.vertices[0].x + face.vertices[1].x + face.vertices[2].x) / 3.0,
                (face.vertices[0].y + face.vertices[1].y + face.vertices[2].y) / 3.0,
                (face.vertices[0].z + face.vertices[1].z + face.vertices[2].z) / 3.0
            );

            // Apply rotation to the center point
            // X-axis rotation
            let mut y1 = center.y;
            let mut z1 = center.z;
            let y2 = y1 * cos_x - z1 * sin_x;
            let z2 = y1 * sin_x + z1 * cos_x;

            // Y-axis rotation
            let mut x2 = center.x;
            z1 = z2;
            let x3 = x2 * cos_y + z1 * sin_y;
            let z3 = -x2 * sin_y + z1 * cos_y;

            // Z-axis rotation
            x2 = x3;
            y1 = y2;
            let x4 = x2 * cos_z - y1 * sin_z;
            let y3 = x2 * sin_z + y1 * cos_z;

            let rotated_center = Vec3::new(x4, y3, z3);

            // Return face and its depth (z-coordinate)
            (face.clone(), rotated_center.z)
        }).collect();

        // Sort by z-coordinate (depth)
        sorted_faces.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal).reverse());

        // Now draw the faces from back to front
        for (face, _) in sorted_faces {
            // Transform each vertex of the face
            let transformed_vertices: Vec<Point2> = face.vertices.iter().map(|vertex| {
                // Scale the vertex
                let scaled = Vec3::new(
                    vertex.x * self.size,
                    vertex.y * self.size,
                    vertex.z * self.size
                );

                // Apply rotation transformations
                // X-axis rotation
                let mut y1 = scaled.y;
                let mut z1 = scaled.z;
                let y2 = y1 * cos_x - z1 * sin_x;
                let z2 = y1 * sin_x + z1 * cos_x;

                // Y-axis rotation
                let mut x2 = scaled.x;
                z1 = z2;
                let x3 = x2 * cos_y + z1 * sin_y;
                let z3 = -x2 * sin_y + z1 * cos_y;

                // Z-axis rotation
                x2 = x3;
                y1 = y2;
                let x4 = x2 * cos_z - y1 * sin_z;
                let y3 = x2 * sin_z + y1 * cos_z;

                // Translate to position
                let final_position = Vec3::new(
                    x4 + self.position.x,
                    y3 + self.position.y,
                    z3 + self.position.z
                );

                // Project to 2D with perspective
                final_position.project(0.0)
            }).collect();

            // Calculate lighting factor based on face normal and viewing direction
            // For simplicity, assume viewing direction is (0, 0, 1)
            let _light_direction = Vec3::new(0.0, 0.0, 1.0);

            // Apply rotation to the normal
            let normal = face.normal;

            // X-axis rotation
            let mut y1 = normal.y;
            let mut z1 = normal.z;
            let y2 = y1 * cos_x - z1 * sin_x;
            let z2 = y1 * sin_x + z1 * cos_x;

            // Y-axis rotation
            let mut x2 = normal.x;
            z1 = z2;
            let x3 = x2 * cos_y + z1 * sin_y;
            let z3 = -x2 * sin_y + z1 * cos_y;

            // Z-axis rotation
            x2 = x3;
            y1 = y2;
            let x4 = x2 * cos_z - y1 * sin_z;
            let y3 = x2 * sin_z + y1 * cos_z;

            let rotated_normal = Vec3::new(x4, y3, z3);

            // Calculate dot product for diffuse lighting
            let dot_product = rotated_normal.z;
            let lighting = 0.3 + 0.7 * dot_product.max(0.0);  // 30% ambient + 70% diffuse

            // Apply lighting and audio reactivity to color
            let base_color = face.color;
            let power = audio_data.power.min(1.0);

            // Enhance color based on audio
            let enhanced_color = rgba(
                base_color.red + power * 0.2,
                base_color.green + power * 0.1,
                base_color.blue + power * 0.3,
                base_color.alpha
            );

            // Create reflection effect
            let reflection_color = if !self.reflection_points.is_empty() {
                // Find the closest reflection point for this face
                let face_center = Point2::new(
                    (transformed_vertices[0].x + transformed_vertices[1].x + transformed_vertices[2].x) / 3.0,
                    (transformed_vertices[0].y + transformed_vertices[1].y + transformed_vertices[2].y) / 3.0
                );

                // Find some reflection points to sample based on face position
                let mut reflection_colors = Vec::new();

                for point in &self.reflection_points {
                    // Calculate distance
                    let dx = point.x - face_center.x;
                    let dy = point.y - face_center.y;
                    let distance = (dx * dx + dy * dy).sqrt();

                    // Only consider points within a certain range
                    if distance < 300.0 {
                        // Weight by inverse distance
                        let weight = 1.0 / (1.0 + distance * 0.01);

                        // Add reflection contribution
                        reflection_colors.push((
                            hsl(
                                (self.rotation[1] * 0.1 + distance * 0.001) % 1.0,
                                0.7,
                                0.5 * weight
                            ),
                            weight
                        ));
                    }
                }

                // Blend reflection colors if we have any
                if !reflection_colors.is_empty() {
                    let total_weight = reflection_colors.iter().map(|(_, w)| w).sum::<f32>();

                    let mut r = 0.0;
                    let mut g = 0.0;
                    let mut b = 0.0;

                    for (color, weight) in reflection_colors {
                        let normalized_weight = weight / total_weight;
                        // Convert HSL to RGB components
                        let rgb = rgb_from_hsl(color.hue.into(), color.saturation, color.lightness);
                        r += rgb.0 * normalized_weight;
                        g += rgb.1 * normalized_weight;
                        b += rgb.2 * normalized_weight;
                    }

                    rgba(r, g, b, 0.7)
                } else {
                    // Default reflection if no points found
                    rgba(0.5, 0.5, 0.5, 0.3)
                }
            } else {
                // Default reflection if no points
                rgba(0.5, 0.5, 0.5, 0.3)
            };

            // Blend base color with reflection
            let final_color = rgba(
                enhanced_color.red * (1.0 - MERKABA_REFLECTION_STRENGTH) + reflection_color.red * MERKABA_REFLECTION_STRENGTH,
                enhanced_color.green * (1.0 - MERKABA_REFLECTION_STRENGTH) + reflection_color.green * MERKABA_REFLECTION_STRENGTH,
                enhanced_color.blue * (1.0 - MERKABA_REFLECTION_STRENGTH) + reflection_color.blue * MERKABA_REFLECTION_STRENGTH,
                enhanced_color.alpha
            );

            // Apply lighting
            let lit_color = rgba(
                final_color.red * lighting,
                final_color.green * lighting,
                final_color.blue * lighting,
                final_color.alpha
            );

            // Draw the face with transformed vertices (clone to avoid move)
            let vertices_for_polygon = transformed_vertices.clone();
            draw.polygon()
                .points(vertices_for_polygon)
                .color(lit_color);

            // Draw individual lines for the triangle outline
            if transformed_vertices.len() >= 3 {
                for i in 0..3 {
                    let start = transformed_vertices[i];
                    let end = transformed_vertices[(i + 1) % 3];

                    draw.line()
                        .start(start)
                        .end(end)
                        .weight(1.0)
                        .color(rgba(1.0, 1.0, 1.0, 0.3));
                }
            }
        }

        // Add a subtle glow to the center for extra effect
        let glow_size = self.size * (1.0 + audio_data.power * 0.5);
        draw.ellipse()
            .x_y(0.0, 0.0)
            .radius(glow_size * 0.7)
            .color(rgba(0.6, 0.6, 1.0, 0.1));
    }
}

// Helper function to convert HSL to RGB
fn rgb_from_hsl(h: f32, s: f32, l: f32) -> (f32, f32, f32) {
    if s == 0.0 {
        // achromatic (grey)
        return (l, l, l);
    }

    let h = h % 1.0; // normalize h to 0-1
    let h = h * 6.0; // sector 0 to 5

    let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
    let x = c * (1.0 - ((h % 2.0) - 1.0).abs());
    let m = l - c/2.0;

    let (r, g, b) = match h as i32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        5 => (c, 0.0, x),
        _ => (0.0, 0.0, 0.0),
    };

    (r + m, g + m, b + m)
}

// Star field particle
struct Star {
    x: f32,
    y: f32,
    z: f32,
    size: f32,
    color: Hsla,
    music_influence: f32,  // How much this star is influenced by music (0-1)
}

// Star field for 3D background
struct StarField {
    stars: Vec<Star>,
    time: f32,
}

impl StarField {
    fn new() -> Self {
        let mut rng = rand::thread_rng();
        let mut stars = Vec::with_capacity(MAX_STARS);

        for _ in 0..MAX_STARS {
            // Create random positions throughout 3D space
            let x = rng.gen_range(-(WINDOW_WIDTH as f32)..WINDOW_WIDTH as f32);
            let y = rng.gen_range(-(WINDOW_HEIGHT as f32)..WINDOW_HEIGHT as f32);
            let z = rng.gen_range(0.0..STAR_FIELD_DEPTH);

            // Base size inversely proportional to starting depth
            let base_size = rng.gen_range(0.5..3.0);

            // Random hue with high brightness
            let hue = rng.gen_range(0.0..1.0);
            let brightness = rng.gen_range(0.7..1.0);

            // How much this star responds to music
            let music_influence = rng.gen_range(0.0..1.0);

            stars.push(Star {
                x,
                y,
                z,
                size: base_size,
                color: hsla(hue, 0.2, brightness, 0.9),
                music_influence,
            });
        }

        Self {
            stars,
            time: 0.0,
        }
    }

    fn update(&mut self, delta_time: f32, audio_data: &AudioData) {
        self.time += delta_time;

        // Calculate overall audio reactivity factor from power
        let overall_reactivity = audio_data.power.min(1.0);

        // Get beat detection from low frequencies
        let bass_power = audio_data.spectrum.iter()
            .take(SECTION_SIZE / 4)  // Use first quarter of a section for bass
            .sum::<f32>() / (SECTION_SIZE / 4) as f32;

        // Get some mid-range frequencies for color shifts
        let mid_power = audio_data.spectrum.iter()
            .skip(SECTION_SIZE / 2)
            .take(SECTION_SIZE / 2)
            .sum::<f32>() / (SECTION_SIZE / 2) as f32;

        let mut rng = rand::thread_rng();

        for star in &mut self.stars {
            // Base speed plus music influence
            let speed_factor = STAR_SPEED_BASE +
                STAR_SPEED_MUSIC_FACTOR * overall_reactivity * star.music_influence;

            // Move star toward viewer (decreasing z)
            star.z -= speed_factor * delta_time;

            // When star passes viewer, reset it to far depth with new random position
            if star.z <= 0.0 {
                star.z = STAR_FIELD_DEPTH;
                star.x = rng.gen_range(-(WINDOW_WIDTH as f32)..WINDOW_WIDTH as f32);
                star.y = rng.gen_range(-(WINDOW_HEIGHT as f32)..WINDOW_HEIGHT as f32);

                // Update color based on current music
                let hue_shift = (self.time * 0.1 + mid_power).fract();
                let brightness = 0.7 + 0.3 * overall_reactivity;
                star.color = hsla(hue_shift, 0.2, brightness, 0.9);

                // Size influenced by bass for 3D pulse effect
                star.size = rng.gen_range(0.5..3.0) * (1.0 + bass_power);

                // Update music influence
                star.music_influence = rng.gen_range(0.2..1.0);
            }

            // Apply subtle swirl effect based on audio
            let swirl_factor = 0.1 * overall_reactivity;
            let angle = self.time * 0.2 * star.music_influence;
            star.x += (angle.sin() * swirl_factor * delta_time * 60.0) * (STAR_FIELD_DEPTH - star.z) / STAR_FIELD_DEPTH;
            star.y += (angle.cos() * swirl_factor * delta_time * 60.0) * (STAR_FIELD_DEPTH - star.z) / STAR_FIELD_DEPTH;
        }
    }

    fn draw(&self, draw: &Draw, audio_data: &AudioData) {
        // Star field should be drawn before other elements
        for star in &self.stars {
            // Calculate perspective projection
            let z_factor = (STAR_FIELD_DEPTH - star.z) / STAR_FIELD_DEPTH;

            // Project 3D to 2D space
            let projected_x = star.x * z_factor;
            let projected_y = star.y * z_factor;

            // Size increases as star gets closer (z decreases)
            let perspective_size = star.size * z_factor * z_factor * 12.0;

            // Alpha increases as star gets closer
            let alpha = star.color.alpha * z_factor;
            let color = hsla(
                star.color.hue.into(), // Convert RgbHue to f32
                star.color.saturation,
                star.color.lightness,
                alpha
            );

            // Add audio-reactive twinkle effect
            let twinkle_factor = 1.0 + audio_data.power * (star.music_influence * 0.5).sin() * 0.5;

            // Draw the star with soft glow
            draw.ellipse()
                .x_y(projected_x, projected_y)
                .radius(perspective_size * twinkle_factor)
                .color(color);

            // Add an inner brighter core for more realistic star appearance
            if perspective_size > 1.0 {
                draw.ellipse()
                    .x_y(projected_x, projected_y)
                    .radius(perspective_size * 0.5)
                    .color(hsla(
                        star.color.hue.into(), // Convert RgbHue to f32
                        star.color.saturation * 0.5,
                        star.color.lightness + 0.2,
                        alpha + 0.2
                    ));
            }
        }
    }
}

// Audio data (spectrum and metadata)
struct AudioData {
    spectrum: Vec<f32>,
    raw_samples: Vec<f32>,
    power: f32,
}

// Song metadata
struct SongMetadata {
    path: PathBuf,
    title: String,
    artist: String,
    display_name: String,
}

impl AudioData {
    fn new() -> Self {
        Self {
            spectrum: vec![0.0; NUM_BANDS],
            raw_samples: Vec::with_capacity(FFT_SIZE),
            power: 0.0,
        }
    }

    fn update(&mut self, spectrum: Vec<f32>, raw_samples: Vec<f32>, power: f32) {
        self.spectrum = spectrum;
        self.raw_samples = raw_samples;
        self.power = power;
    }
}

impl SongMetadata {
    fn new(path: PathBuf) -> Self {
        let file_name = path.file_name()
            .and_then(|f| f.to_str())
            .unwrap_or("Unknown")
            .to_string();

        // Default to file name
        let mut title = file_name.clone();
        let mut artist = String::new();

        // Try to read ID3 tags for MP3 files
        if let Some(ext) = path.extension() {
            if ext == "mp3" {
                if let Ok(tag) = Tag::read_from_path(&path) {
                    // Get title from ID3 tag
                    if let Some(tag_title) = tag.title() {
                        if !tag_title.trim().is_empty() {
                            title = tag_title.to_string();
                        }
                    }

                    // Get artist from ID3 tag
                    if let Some(tag_artist) = tag.artist() {
                        if !tag_artist.trim().is_empty() {
                            artist = tag_artist.to_string();
                        }
                    }
                }
            }
        }

        // Create a display name with artist - title format
        let display_name = if !artist.is_empty() {
            format!("{} - {}", artist, title)
        } else {
            title.clone()
        };

        Self {
            path,
            title,
            artist,
            display_name,
        }
    }
}

struct Model {
    // Audio
    _stream_handle: rodio::OutputStreamHandle,
    sink: Sink,
    _output_stream: rodio::OutputStream,  // Keep this to prevent destruction
    audio_data: Arc<Mutex<AudioData>>,
    current_song: String,
    is_paused: Arc<AtomicBool>,

    // Song history
    song_history: Vec<PathBuf>,  // History of played songs
    history_position: usize,     // Current position in song history

    // Song picker
    show_song_picker: bool,      // Whether the song picker is visible
    all_songs: Vec<SongMetadata>,     // All available songs
    selected_song_index: usize,  // Currently selected song in the picker
    song_picker_scroll: usize,   // Scroll position for song list
    song_hover_index: Option<usize>, // Currently hovered song index
    last_click_time: Instant,    // For detecting double clicks

    // Audio processing
    audio_receiver: mpsc::Receiver<Vec<f32>>,

    // Visualization
    visualizer: SpectrumVisualizer,
    star_field: StarField,  // Add star field
    merkaba: Merkaba,  // Add merkaba to model

    // UI state
    show_fps: bool,
    hide_ui: bool,          // Whether to hide UI elements (text and controls)
    last_fps_update: Instant,
    frame_count: u32,
    fps: f32,

    // Mouse interaction
    mouse_position: Point2,
    mouse_influence: f32,  // 0.0 = no influence, 1.0 = full influence

    // FFT processing
    fft_planner: FftPlanner<f32>,
    window_function: Vec<f32>,

    // Volume control
    volume: f32,
    volume_changed_time: Instant,
    show_volume_indicator: bool,
}

// Audio reactive visualization
struct SpectrumVisualizer {
    time: f32,  // For time-based animations
    history: Vec<Vec<(f32, f32)>>,  // History of positions for each band
    amplitude_history: Vec<Vec<f32>>,  // History of amplitudes for each band for animation
    target_radii: Vec<f32>,  // Target radius for each band
    current_radii: Vec<f32>,  // Current radius for each band
    inner_radii: Vec<f32>,    // Dynamic inner radius for each band
    velocity: Vec<f32>,       // Velocity of amplitude change for each band
    mouse_influenced_positions: Vec<(f32, f32)>, // Positions influenced by mouse
}

// Easing functions for smooth animations
fn ease_in_out_cubic(t: f32) -> f32 {
    if t < 0.5 {
        4.0 * t * t * t
    } else {
        1.0 - (-2.0 * t + 2.0).powf(3.0) / 2.0
    }
}

fn ease_out_elastic(t: f32) -> f32 {
    let c4 = (2.0 * PI) / 3.0;
    if t == 0.0 {
        0.0
    } else if t == 1.0 {
        1.0
    } else {
        2.0_f32.powf(-10.0 * t) * ((t * 10.0 - 0.75) * c4).sin() + 1.0
    }
}

fn ease_in_out_quad(t: f32) -> f32 {
    if t < 0.5 {
        2.0 * t * t
    } else {
        1.0 - (-2.0 * t + 2.0).powf(2.0) / 2.0
    }
}

impl SpectrumVisualizer {
    fn new() -> Self {
        let mut history = Vec::with_capacity(NUM_BANDS);
        let mut amplitude_history = Vec::with_capacity(NUM_BANDS);
        let target_radii = vec![CIRCLE_RADIUS; NUM_BANDS];
        let current_radii = vec![CIRCLE_RADIUS; NUM_BANDS];
        let inner_radii = vec![CIRCLE_RADIUS; NUM_BANDS];
        let velocity = vec![0.0; NUM_BANDS];
        let mouse_influenced_positions = vec![(0.0, 0.0); NUM_BANDS];

        for _ in 0..NUM_BANDS {
            history.push(Vec::with_capacity(TRACER_LENGTH));
            amplitude_history.push(vec![0.0; 3]); // Store a few frames of amplitude history
        }

        Self {
            time: 0.0,
            history,
            amplitude_history,
            target_radii,
            current_radii,
            inner_radii,
            velocity,
            mouse_influenced_positions,
        }
    }

    fn update(&mut self, delta_time: f32, audio_data: &AudioData, mouse_pos: Point2, mouse_influence: f32) {
        // Update time for color animation
        self.time += delta_time;

        let center_x = 0.0; // In Nannou, (0,0) is the center of the window
        let center_y = 0.0;

        // Calculate and store new positions
        for i in 0..NUM_BANDS {
            let angle = (i as f32 / NUM_BANDS as f32) * 2.0 * PI + ROTATION_OFFSET;

            // Apply audio data to radius - use direct amplitude, no extra scaling
            let amplitude = audio_data.spectrum[i];

            // Update amplitude history (shift values and add new one)
            self.amplitude_history[i].remove(0);
            self.amplitude_history[i].push(amplitude);

            // Get previous amplitude for comparison
            let prev_amplitude = self.amplitude_history[i][1];

            // Calculate velocity of amplitude change
            self.velocity[i] = (amplitude - prev_amplitude) / delta_time;

            // Apply more aggressive easing to amplitude for more responsive animation
            let eased_amplitude = amplitude * 0.7 + ease_in_out_quad(amplitude) * 0.3; // Modified from 0.9/0.1 to 0.7/0.3

            // Enhanced breathing effect (increased from 2.0 to 6.0)
            // Apply consistent breathing with symmetry by using angle
            let phase_offset = (i as f32 / HALF_BANDS as f32) * PI; // Create a phase offset based on position
            let time_factor = (self.time * 0.5 + phase_offset).sin();
            let breathing = ease_in_out_cubic(time_factor.abs()) * 6.0 * eased_amplitude;

            // Calculate target radius with easing applied for smoother transitions
            let target_radius = CIRCLE_RADIUS + eased_amplitude * MAX_AMPLITUDE + breathing;
            self.target_radii[i] = target_radius;

            // Calculate current radius with drop behavior:
            // If amplitude is decreasing, drop toward center at DROP_SPEED
            // If amplitude is increasing or steady, ease back to target at RISE_SPEED
            if amplitude < prev_amplitude {
                // Drop toward center - faster when bigger drops in amplitude
                let drop_factor = (prev_amplitude - amplitude) * DROP_SPEED;

                // Calculate a dynamic speed modifier based on distance from center
                // Further out = slower movement (apply OUTWARD_SLOWDOWN_FACTOR)
                let extension_ratio = (self.current_radii[i] - CIRCLE_RADIUS) / MAX_AMPLITUDE;
                let speed_modifier = 1.0 - (extension_ratio * OUTWARD_SLOWDOWN_FACTOR);

                self.current_radii[i] -= drop_factor * delta_time * 60.0 * speed_modifier;

                // Allows dropping to 30% of base radius (more dramatic drops)
                self.current_radii[i] = self.current_radii[i].max(CIRCLE_RADIUS * 0.3);
            } else {
                // Rise toward target radius with easing
                let distance = self.target_radii[i] - self.current_radii[i];

                // Calculate dynamic speed modifier - closer to center = faster movement
                let extension_ratio = (self.current_radii[i] - CIRCLE_RADIUS) / MAX_AMPLITUDE;
                let center_ratio = 1.0 - extension_ratio;
                // Apply INWARD_SPEEDUP_FACTOR when closer to center
                let speed_modifier = 1.0 + (center_ratio * INWARD_SPEEDUP_FACTOR);

                let rise_amount = distance * RISE_SPEED * delta_time * speed_modifier;
                self.current_radii[i] += rise_amount;
            }

            // Update inner radius based on amplitude velocity
            // Negative velocity (amplitude decreasing) causes inner radius to contract
            // The faster the drop, the more contraction
            let velocity_factor = self.velocity[i].abs().min(5.0) / 5.0;  // Normalize to 0-1 range
            let contraction = if self.velocity[i] < 0.0 {
                // Amplitude is decreasing - contract inner circle
                velocity_factor * INNER_RADIUS_DROP_FACTOR * (CIRCLE_RADIUS - MIN_INNER_RADIUS)
            } else {
                // Amplitude is increasing - expand inner circle
                0.0
            };

            // Calculate target inner radius with contraction
            let target_inner = CIRCLE_RADIUS - contraction;

            // Smoothly approach target inner radius
            self.inner_radii[i] += (target_inner - self.inner_radii[i]) * 10.0 * delta_time;

            // Calculate normal position based on radius
            let radius = self.current_radii[i];
            let normal_x = center_x + radius * angle.cos();
            let normal_y = center_y + radius * angle.sin();

            // Store normal position in mouse_influenced_positions
            self.mouse_influenced_positions[i] = (normal_x, normal_y);

            // Apply mouse influence if active
            if mouse_influence > 0.0 {
                // Calculate distance from this point to the mouse
                let mouse_distance = ((normal_x - mouse_pos.x).powi(2) + (normal_y - mouse_pos.y).powi(2)).sqrt();

                // Define the radius of influence (how far the mouse affects points)
                let influence_radius = 450.0;  // Increased from 150.0 to 450.0 (3x)

                // Only apply influence if mouse is within range
                if mouse_distance < influence_radius {
                    // Calculate influence factor (stronger when mouse is closer)
                    let distance_factor = 1.0 - (mouse_distance / influence_radius).min(1.0);
                    let smooth_factor = ease_in_out_cubic(distance_factor);

                    // Direction to/from mouse
                    let dir_x = mouse_pos.x - normal_x;
                    let dir_y = mouse_pos.y - normal_y;
                    let dir_length = (dir_x.powi(2) + dir_y.powi(2)).sqrt().max(0.1);

                    // Normalize direction vector
                    let normalized_dir_x = dir_x / dir_length;
                    let normalized_dir_y = dir_y / dir_length;

                    // Calculate attraction/repulsion force
                    // Positive values attract, negative values repel
                    let force = 40.0;

                    // Apply mouse influence to position
                    let influenced_x = normal_x + normalized_dir_x * force * smooth_factor * mouse_influence;
                    let influenced_y = normal_y + normalized_dir_y * force * smooth_factor * mouse_influence;

                    // Update the influenced position
                    self.mouse_influenced_positions[i] = (influenced_x, influenced_y);
                }
            }

            // Add to history - using mouse influenced position
            if self.history[i].len() >= TRACER_LENGTH {
                self.history[i].remove(0); // Remove oldest position
            }
            self.history[i].push(self.mouse_influenced_positions[i]);
        }
    }

    fn draw(&self, draw: &Draw, audio_data: &AudioData) {
        // Create a richer background with subtle gradients
        draw.background().color(rgba(0.02, 0.02, 0.05, 1.0));

        // First draw a dynamic inner circle based on average inner radius
        let avg_inner_radius = self.inner_radii.iter().sum::<f32>() / self.inner_radii.len() as f32;
        draw.ellipse()
            .x_y(0.0, 0.0)
            .radius(avg_inner_radius * 0.95)
            .color(rgba(0.05, 0.05, 0.1, 0.2));

        // First draw a subtle background glow
        draw.ellipse()
            .x_y(0.0, 0.0)
            .radius(CIRCLE_RADIUS * 0.95)
            .color(rgba(0.05, 0.05, 0.1, 0.2));

        // Calculate the average amplitude for the binding line
        // Use more points (240 instead of 180) and a more direct mapping
        let mut avg_points = Vec::with_capacity(240);
        for i in 0..240 {
            let angle = (i as f32 / 240.0) * 2.0 * PI + ROTATION_OFFSET;

            // Take exactly 1-2 bands for each point rather than a range
            // This prevents "chunks" moving together
            let band_idx = ((i as f32 / 240.0) * NUM_BANDS as f32) as usize % NUM_BANDS;

            let amplitude = if !self.history[band_idx].is_empty() {
                audio_data.spectrum[band_idx]
            } else {
                0.0
            };

            // Milder easing for the binding line
            let eased_amplitude = amplitude * 0.7 + ease_in_out_quad(amplitude) * 0.3;

            // Calculate radius with less exaggerated smoothing
            let avg_radius = CIRCLE_RADIUS + eased_amplitude * MAX_AMPLITUDE * 0.75;

            // Calculate position
            let x = avg_radius * angle.cos();
            let y = avg_radius * angle.sin();

            avg_points.push((x, y, amplitude));
        }

        // Draw the binding ring - first draw a slightly wider, more transparent version for a glow effect
        if avg_points.len() > 2 {
            // Create a path with all points for glow effect
            let glow_path_points: Vec<Point2> = avg_points.iter()
                .map(|(x, y, _)| Point2::new(*x, *y))
                .collect();

            // Calculate the base color with animated hue
            let time_shift = self.time * 5.0;
            let base_hue = (time_shift * 0.1) % 1.0;

            // Draw the glow path
            draw.polygon()
                .stroke_weight(8.0) // Wider stroke for glow
                .stroke(hsla(base_hue, 0.7, 0.5, 0.2)) // More transparent
                .no_fill()
                .points(glow_path_points.clone());

            // Draw the main line with gradient effect
            for i in 0..avg_points.len() - 1 {
                let (x1, y1, amp1) = avg_points[i];
                let (x2, y2, amp2) = avg_points[i + 1];

                // Calculate section and position similar to the tracer coloring
                let binding_sections = 12; // Same as our SECTIONS constant
                let binding_section_size = avg_points.len() / binding_sections;

                let section = i / binding_section_size;
                let pos_in_section = i % binding_section_size;

                // Normalize position within section (0 to 1)
                let section_normalized_pos = pos_in_section as f32 / binding_section_size as f32;

                // Every odd section should have reversed coloring
                let section_pos = if section % 2 == 0 {
                    section_normalized_pos
                } else {
                    1.0 - section_normalized_pos
                };

                // Calculate which spike this belongs to
                let spike = section / 2;

                // Animate the hue to create a shifting rainbow effect with symmetry within spikes
                let base_hue = (spike as f32 / (binding_sections / 2) as f32 + base_hue) % 1.0;
                let segment_hue = (base_hue + section_pos * 0.1) % 1.0;

                // Adjust brightness and saturation based on amplitude
                let avg_amp = (amp1 + amp2) * 0.5;
                let brightness = 0.5 + avg_amp * 0.5; // 50-100% brightness
                let saturation = 0.7 + avg_amp * 0.3; // 70-100% saturation

                // Draw this segment with the calculated properties
                draw.line()
                    .start(Point2::new(x1, y1))
                    .end(Point2::new(x2, y2))
                    .weight(3.0 + avg_amp * 3.0) // Variable line thickness based on amplitude
                    .color(hsla(segment_hue, saturation, brightness, 0.8));
            }

            // Close the loop by connecting the last point to the first
            let (x1, y1, amp1) = avg_points[avg_points.len() - 1];
            let (x2, y2, amp2) = avg_points[0];

            let segment_hue = (base_hue + 0.9) % 1.0;
            let avg_amp = (amp1 + amp2) * 0.5;
            let brightness = 0.5 + avg_amp * 0.5;
            let saturation = 0.7 + avg_amp * 0.3;

            draw.line()
                .start(Point2::new(x1, y1))
                .end(Point2::new(x2, y2))
                .weight(3.0 + avg_amp * 3.0)
                .color(hsla(segment_hue, saturation, brightness, 0.8));

            // Add subtle pulsing points at amplitude peaks
            for (i, (x, y, amp)) in avg_points.iter().enumerate() {
                if *amp > 0.6 {
                    // Check if this is a local maximum
                    let prev = if i > 0 { avg_points[i-1].2 } else { avg_points[avg_points.len()-1].2 };
                    let next = if i < avg_points.len()-1 { avg_points[i+1].2 } else { avg_points[0].2 };

                    if *amp > prev && *amp > next {
                        // This is a peak, add a glowing point
                        // Use the same symmetric color system
                        let normalized_pos = (i as f32 / avg_points.len() as f32) * 2.0; // 0.0 to 2.0
                        let mirrored_pos = if normalized_pos < 1.0 { normalized_pos } else { 2.0 - normalized_pos };
                        let point_hue = (base_hue + mirrored_pos * 0.3 + 0.1) % 1.0;

                        let pulse_size = 3.0 + (*amp - 0.6) * 10.0;
                        let pulse_alpha = 0.6 + (*amp - 0.6) * 0.4;

                        draw.ellipse()
                            .x_y(*x, *y)
                            .radius(pulse_size)
                            .color(hsla(point_hue, 1.0, 0.7, pulse_alpha));
                    }
                }
            }
        }

        // Draw tracers with high-quality anti-aliasing
        for i in 0..NUM_BANDS {
            // Calculate which section this band belongs to
            let section = i / SECTION_SIZE;
            let pos_in_section = i % SECTION_SIZE;

            // Normalize position within section (0 to 1)
            let section_normalized_pos = pos_in_section as f32 / SECTION_SIZE as f32;

            // Every odd section should have reversed coloring
            let section_pos = if section % 2 == 0 {
                section_normalized_pos
            } else {
                1.0 - section_normalized_pos
            };

            // Calculate which spike this belongs to
            let spike = section / 2;

            // Calculate hue based on spike and position within spike
            // Each spike gets its own color range
            let base_hue = (spike as f32 / SPIKES as f32) * 360.0;
            let pos_hue = section_pos * 60.0; // 60 degree range within each spike

            let time_shift = self.time * 10.0;
            let hue = (base_hue + pos_hue + time_shift) % 360.0;

            // Draw tracer history
            let history_len = self.history[i].len();

            if history_len >= 3 {
                // Instead of drawing individual line segments, draw a path with all points
                let points = self.history[i].iter()
                    .enumerate()
                    .map(|(j, &(x, y))| {
                        // Apply easing function to age factor for smoother fade-out
                        let raw_age_factor = j as f32 / (history_len - 1) as f32;
                        let age_factor = ease_in_out_cubic(raw_age_factor);

                        // Enhanced color calculation with saturation variation
                        let brightness = 0.3 + 0.7 * age_factor; // 30% to 100% brightness
                        let alpha = ease_in_out_quad(raw_age_factor); // Smoothed opacity transition

                        // Saturation with subtle pulsing
                        let saturation_pulse = (self.time + i as f32 * 0.01).sin() * 0.1 + 0.9;
                        let saturation = 0.8 + 0.2 * saturation_pulse;

                        // Add a slight hue shift along the trail for rainbow effect with easing
                        let hue_shift = 20.0 * (1.0 - ease_out_elastic(raw_age_factor));
                        let point_hue = (hue - hue_shift) % 360.0;

                        // Weight based on position and audio reactivity with easing
                        let amplitude = audio_data.spectrum[i];
                        let eased_amplitude = ease_in_out_quad(amplitude);

                        // Thinner lines overall - REDUCED from 3.0 to 2.0
                        let weight = (1.0 + eased_amplitude * 2.0) * (0.2 + 0.8 * age_factor);

                        (Point2::new(x, y), point_hue / 360.0, saturation, brightness, alpha, weight)
                    })
                    .collect::<Vec<_>>();

                // Get inner point coordinates
                let angle = (i as f32 / NUM_BANDS as f32) * 2.0 * PI + ROTATION_OFFSET;
                let inner_x = self.inner_radii[i] * angle.cos();
                let inner_y = self.inner_radii[i] * angle.sin();

                // Draw connecting line from inner circle to first point of tracer
                if !points.is_empty() {
                    let (first_point, first_hue, first_sat, first_brightness, first_alpha, _) = points[0];

                    // Create gradient for inner connector
                    draw.line()
                        .start(Point2::new(inner_x, inner_y))
                        .end(first_point)
                        .weight(1.0)
                        .color(hsla(
                            first_hue,
                            first_sat * 0.7,
                            first_brightness * 0.7,
                            first_alpha * 0.5,
                        ));
                }

                // Draw the path as a polyline with gradient effect
                for j in 0..points.len()-1 {
                    let (p1, hue1, sat1, brightness1, alpha1, weight1) = points[j];
                    let (p2, hue2, sat2, brightness2, alpha2, weight2) = points[j+1];

                    // Use average values for the line
                    let color = hsla(
                        (hue1 + hue2) * 0.5,
                        (sat1 + sat2) * 0.5,
                        (brightness1 + brightness2) * 0.5,
                        (alpha1 + alpha2) * 0.5,
                    );

                    // Draw this segment with the calculated properties
                    draw.line()
                        .start(p1)
                        .end(p2)
                        .weight((weight1 + weight2) * 0.5)
                        .color(color);
                }
            }

            // Draw the current point with enhanced glow effect and growing animation
            if !self.history[i].is_empty() {
                let (x, y) = *self.history[i].last().unwrap();
                let amplitude = audio_data.spectrum[i];

                // Reduce growth factor calculations - make them more subtle
                let prev_amp = self.amplitude_history[i][1];
                let growth_factor = if amplitude > prev_amp {
                    // Growing phase - more responsive (increased from 0.6 to 1.2)
                    let normalized_change = ((amplitude - prev_amp) / amplitude).min(1.0) * 1.2;
                    ease_out_elastic(normalized_change) * 1.2 + 0.3 // Increased from 0.8/0.2 to 1.2/0.3
                } else {
                    // Shrinking phase - more responsive (increased from 0.3 to 0.5)
                    1.0 - ease_in_out_cubic(((prev_amp - amplitude) / prev_amp).min(1.0)) * 0.5
                };

                // More aggressive easing for amplitude
                let eased_amplitude = amplitude * 0.6 + ease_in_out_quad(amplitude) * 0.4; // Changed from 0.7/0.3 to 0.6/0.4

                // Enhanced glow effects
                let glow_intensity = 0.6 + eased_amplitude * 0.4; // Increased from 0.2 to 0.4

                // Larger base size for glows
                let base_glow_size = 2.0 + eased_amplitude * 5.0; // Increased from 1.5/3.0 to 2.0/5.0
                let glow_size = base_glow_size * growth_factor;

                // Main glow point using the symmetric hue
                let main_color = hsla(
                    hue / 360.0,
                    1.0,
                    glow_intensity,
                    1.0,
                );

                // Inner glow (brighter)
                draw.ellipse()
                    .x_y(x, y)
                    .radius(glow_size * 0.4)
                    .color(main_color);

                // Outer glow (more transparent)
                draw.ellipse()
                    .x_y(x, y)
                    .radius(glow_size * 0.8)
                    .color(hsla(
                        hue / 360.0,
                        0.9,
                        glow_intensity * 0.8,
                        0.4, // REDUCED from 0.5 to 0.4
                    ));

                // Extra outer glow - LOWERED threshold from 0.5 to 0.3 to make it happen more often
                if amplitude > 0.3 {
                    // Detect sudden increases for more dramatic pops
                    let sudden_increase = amplitude > prev_amp * 1.1; // More sensitive (1.2 -> 1.1)

                    // More dramatic pop effect - INCREASED factors
                    let pop_factor = if sudden_increase {
                        ease_out_elastic(amplitude - 0.3) * 3.5 // INCREASED from 2.5 to 3.5
                    } else {
                        ease_in_out_cubic(amplitude - 0.3) * 1.8 // INCREASED from 1.2 to 1.8
                    };

                    draw.ellipse()
                        .x_y(x, y)
                        .radius(glow_size * (1.0 + pop_factor))
                        .color(hsla(
                            hue / 360.0,
                            0.8,
                            glow_intensity * 0.6,
                            0.2, // REDUCED from 0.3 to 0.2
                        ));
                }
            }
        }

        // Add a more intense overall glow
        let power = audio_data.power.min(1.0);
        let eased_power = ease_in_out_cubic(power);
        // LOWERED threshold to make it happen more often
        if eased_power > 0.1 {
            // Pulsing effect - INCREASED factors
            let pulse = (self.time * 3.0).sin() * 0.25 + 1.0; // INCREASED animation speed and range
            draw.ellipse()
                .x_y(0.0, 0.0)
                .radius(CIRCLE_RADIUS * (0.85 + eased_power * 0.6 * pulse)) // Increased from 0.4 to 0.6
                .color(rgba(eased_power * 0.4, eased_power * 0.2, eased_power * 0.5, eased_power * 0.25)); // INCREASED color intensity
        }
    }
}

// Gets a random music file from the music directory
fn get_random_music_file() -> Option<PathBuf> {
    let songs = get_all_music_files();

    if songs.is_empty() {
        println!("No music files found in directory: {}", MUSIC_DIR);
        return None;
    }

    let selected = songs.choose(&mut rand::thread_rng()).map(|song| song.path.clone());
    if let Some(path) = &selected {
        println!("Selected music file: {:?}", path);
    }
    selected
}

// Play a music file - modified to capture audio for visualization
fn play_music(sink: &Sink, file_path: &std::path::Path, audio_sender: mpsc::Sender<Vec<f32>>) -> Option<String> {
    // Open file
    let file = match std::fs::File::open(file_path) {
        Ok(file) => file,
        Err(e) => {
            println!("Failed to open music file: {:?}", e);
            return None;
        }
    };

    // Create a BufReader for the file
    let reader = BufReader::new(file);

    // Decode the audio file
    let source = match Decoder::new(reader) {
        Ok(source) => source,
        Err(e) => {
            println!("Failed to decode music file: {:?}", e);
            return None;
        }
    };

    // Get audio properties
    let channels = source.channels();
    let sample_rate = source.sample_rate();

    println!("Audio info: {} channels, {} Hz sample rate", channels, sample_rate);

    // Open the file again for visualization
    // We need a second file reader because we can't clone the decoder
    if let Ok(visualization_file) = std::fs::File::open(file_path) {
        let visualization_reader = BufReader::new(visualization_file);
        if let Ok(visualization_source) = Decoder::new(visualization_reader) {
            // Spawn a thread to feed audio data to the visualizer
            let audio_sender_clone = audio_sender.clone();
            let is_stereo = channels == 2;

            // Calculate more accurate sleep time based on sample rate
            // Formula: (buffer_size / sample_rate) * 1000 = milliseconds per buffer
            let buffer_size = 1024;
            let sleep_time_ms = ((buffer_size as f32 / sample_rate as f32) * 1000.0) as u64;

            // Apply a slight adjustment factor to reduce perceived lag (0.8 = 20% faster)
            let adjusted_sleep_time = (sleep_time_ms as f32 * 0.8) as u64;

            println!("Audio buffer: {} samples, delay: {}ms (adjusted: {}ms)",
                    buffer_size, sleep_time_ms, adjusted_sleep_time);

            thread::spawn(move || {
                // Process the entire audio file
                let mut buffer = Vec::with_capacity(buffer_size);
                let mut temp_buffer = Vec::with_capacity(buffer_size);

                // This works like an iterator, collecting samples from the visualization source
                for sample in visualization_source.convert_samples() {
                    temp_buffer.push(sample);

                    // Process in chunks of buffer_size samples (or equivalent for stereo)
                    if temp_buffer.len() >= buffer_size * channels as usize {
                        buffer.clear();

                        // Convert to mono if stereo by averaging channels
                        if is_stereo {
                            for chunk in temp_buffer.chunks(2) {
                                if chunk.len() == 2 {
                                    buffer.push((chunk[0] + chunk[1]) * 0.5);
                                }
                            }
                        } else {
                            buffer.extend_from_slice(&temp_buffer);
                        }

                        // Send the buffer for visualization
                        let _ = audio_sender_clone.send(buffer.clone());

                        // Wait based on calculated time to simulate real-time playback
                        thread::sleep(Duration::from_millis(adjusted_sleep_time));

                        temp_buffer.clear();
                    }
                }
            });
        }
    }

    // Create a display name with metadata
    let display_name = get_song_display_name(file_path);

    // Clear previous audio and play the new file
    sink.clear();
    sink.append(source);
    sink.play();

    println!("Now playing: {}", display_name);
    Some(display_name)
}

// Get a nice display name for a song with metadata
fn get_song_display_name(path: &std::path::Path) -> String {
    let file_name = path.file_name()
        .and_then(|f| f.to_str())
        .unwrap_or("Unknown")
        .to_string();

    // Try to read ID3 tags for MP3 files
    if let Some(ext) = path.extension() {
        if ext == "mp3" {
            if let Ok(tag) = Tag::read_from_path(path) {
                let title = tag.title().unwrap_or(&file_name);
                let artist = tag.artist();

                if let Some(artist_name) = artist {
                    if !artist_name.trim().is_empty() {
                        return format!("{} - {}", artist_name, title);
                    }
                }

                return title.to_string();
            }
        }
    }

    // Fallback to filename
    file_name
}

// Play the next (random) song
fn next_song(model: &mut Model) {
    if let Some(music_file) = get_random_music_file() {
        // Add to history before playing
        // If we're in the middle of the history, remove future entries
        if model.history_position < model.song_history.len() {
            model.song_history.truncate(model.history_position + 1);
        }
        model.song_history.push(music_file.clone());
        model.history_position = model.song_history.len() - 1;

        play_selected_song(model, &music_file);
    }
}

// Play the previous song from history
fn previous_song(model: &mut Model) {
    // Check if we have previous songs
    if model.history_position > 0 {
        model.history_position -= 1;
        if let Some(previous_file) = model.song_history.get(model.history_position).cloned() {
            play_selected_song(model, &previous_file);
        }
    } else {
        println!("No previous songs in history");
    }
}

// Common function to play a selected song
fn play_selected_song(model: &mut Model, file_path: &PathBuf) {
    // Create a new sender for this song
    let (temp_sender, temp_receiver) = mpsc::channel::<Vec<f32>>();

    // Replace the model's receiver with the new one
    model.audio_receiver = temp_receiver;

    // Play the music
    if let Some(song_name) = play_music(&model.sink, file_path, temp_sender) {
        model.current_song = song_name;
        model.is_paused.store(false, Ordering::Relaxed);
    }
}

fn main() {
    nannou::app(model)
        .update(update)
        .run();
}

// Process audio data using FFT to get frequency spectrum
fn process_audio_data(model: &mut Model, audio_chunk: Vec<f32>) {
    let mut fft_input = vec![Complex { re: 0.0, im: 0.0 }; FFT_SIZE];

    // Fill FFT input with audio data and apply window function
    for i in 0..audio_chunk.len().min(FFT_SIZE) {
        fft_input[i].re = audio_chunk[i] * model.window_function[i];
    }

    // Create a FFT instance
    let fft = model.fft_planner.plan_fft_forward(FFT_SIZE);

    // Perform FFT
    let mut fft_output = fft_input.clone();
    fft.process(&mut fft_output);

    // Calculate magnitude of each frequency bin
    let mut magnitudes = vec![0.0; FFT_SIZE / 2];
    for i in 0..FFT_SIZE / 2 {
        magnitudes[i] = (fft_output[i].re * fft_output[i].re +
                         fft_output[i].im * fft_output[i].im).sqrt();
    }

    // Map FFT bins to visualization bands using golden ratio distribution
    // This gives more weight to lower frequencies and spreads them across more bands
    let mut spectrum = vec![0.0; NUM_BANDS];

    // Create a single section of processed data to duplicate
    let mut section_data = vec![0.0; SECTION_SIZE];

    // Golden ratio for distributing frequencies
    let phi = 1.618033988749895;

    // Process a single section of bands
    for i in 0..SECTION_SIZE {
        // Normalize position (0 to 1)
        let normalized_pos = i as f32 / SECTION_SIZE as f32;

        // Apply golden ratio curve to give more weight to lower frequencies
        let curve_factor = normalized_pos.powf(1.0 / phi);

        // Map to frequency bin with golden ratio curve
        let bin_index = (curve_factor * (FFT_SIZE as f32 / 4.0)) as usize;

        // Add some values from neighboring bins for a richer spectrum
        let primary_bin = bin_index.min(FFT_SIZE / 2 - 1);
        let secondary_bin = (primary_bin + 3).min(FFT_SIZE / 2 - 1);
        let tertiary_bin = (primary_bin / 2).max(1);

        // Combine values with higher scaling factor
        let value = (magnitudes[primary_bin] * 0.12 +
                     magnitudes[secondary_bin] * 0.04 +
                     magnitudes[tertiary_bin] * 0.06).min(1.0);

        // Store in section data
        section_data[i] = value;
    }

    // Now duplicate this section data around the circle creating symmetric patterns
    // Every even section gets the original data, every odd section gets the reversed data
    for section in 0..SECTIONS {
        for i in 0..SECTION_SIZE {
            let spectrum_index = section * SECTION_SIZE + i;

            if section % 2 == 0 {
                // Even sections get original data
                spectrum[spectrum_index] = section_data[i];
            } else {
                // Odd sections get reversed data (for symmetry within each spike)
                spectrum[spectrum_index] = section_data[SECTION_SIZE - 1 - i];
            }
        }
    }

    // Apply smoothing between bands to avoid harsh transitions
    // but with a narrower window to preserve the spike pattern
    let mut smoothed = spectrum.clone();
    for i in 1..NUM_BANDS-1 {
        smoothed[i] = spectrum[i-1] * 0.1 + spectrum[i] * 0.8 + spectrum[i+1] * 0.1;
    }

    // Connect the ends with the same smoothing
    smoothed[0] = spectrum[NUM_BANDS-1] * 0.1 + spectrum[0] * 0.8 + spectrum[1] * 0.1;
    smoothed[NUM_BANDS-1] = spectrum[NUM_BANDS-2] * 0.1 + spectrum[NUM_BANDS-1] * 0.8 + spectrum[0] * 0.1;

    // Calculate overall audio power
    let power = audio_chunk.iter().map(|&s| s.abs()).sum::<f32>() / audio_chunk.len() as f32;

    // Update the audio data
    if let Ok(mut data) = model.audio_data.lock() {
        data.update(smoothed, audio_chunk, power);
    }
}

// Get all music files from the music directory
fn get_all_music_files() -> Vec<SongMetadata> {
    let mut songs = match fs::read_dir(MUSIC_DIR) {
        Ok(entries) => entries
            .filter_map(|entry| {
                let entry = entry.ok()?;
                let path = entry.path();
                if path.is_file() && path.extension().map_or(false, |ext|
                    ext == "mp3" || ext == "wav" || ext == "ogg" || ext == "flac") {
                    Some(SongMetadata::new(path))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>(),
        Err(e) => {
            println!("Failed to read music directory: {}, error: {:?}", MUSIC_DIR, e);
            Vec::new()
        }
    };

    // Sort songs alphabetically by display name
    songs.sort_by(|a, b| a.display_name.to_lowercase().cmp(&b.display_name.to_lowercase()));

    songs
}

// Gets a song name from a path
fn get_song_name(path: &PathBuf) -> String {
    path.file_name()
        .and_then(|f| f.to_str())
        .unwrap_or("Unknown")
        .to_string()
}

// Model setup function
fn model(app: &App) -> Model {
    // Create window with title
    app.new_window()
        .size(WINDOW_WIDTH, WINDOW_HEIGHT)
        .title("Nannou Audio Visualizer")
        .key_pressed(key_pressed)
        .mouse_pressed(mouse_pressed)
        .mouse_moved(mouse_moved)
        .mouse_wheel(mouse_wheel)
        .view(view)
        .build()
        .unwrap();

    println!("Setting up audio...");

    // Set up audio output for music playback
    let (output_stream, stream_handle) = OutputStream::try_default()
        .expect("Failed to get default output device");

    let sink = Sink::try_new(&stream_handle)
        .expect("Failed to create audio sink");

    // Load saved settings
    let settings = load_settings();

    // Apply volume from settings
    sink.set_volume(settings.volume);

    // Create shared audio data
    let audio_data = Arc::new(Mutex::new(AudioData::new()));

    // Channel for passing audio data between threads
    let (audio_sender, audio_receiver) = mpsc::channel::<Vec<f32>>();

    // Set up visualizer
    let visualizer = SpectrumVisualizer::new();
    let star_field = StarField::new();  // Initialize star field
    let merkaba = Merkaba::new();  // Initialize merkaba

    // Create Hann window function for FFT
    let window_function: Vec<f32> = (0..FFT_SIZE)
        .map(|i| 0.5 * (1.0 - ((2.0 * PI * i as f32) / FFT_SIZE as f32).cos()))
        .collect();

    // Try to play a music file
    let mut current_song = String::new();
    if let Some(music_file) = get_random_music_file() {
        if let Some(song_name) = play_music(&sink, &music_file, audio_sender.clone()) {
            current_song = song_name;
        }
    }

    // Pause state flag
    let is_paused = Arc::new(AtomicBool::new(false));

    // Initialize with empty song history
    let song_history = Vec::new();
    let history_position = 0;

    // Load all songs
    let all_songs = get_all_music_files();
    let selected_song_index = 0;
    let show_song_picker = false;
    let song_picker_scroll = 0;
    let song_hover_index = None;
    let last_click_time = Instant::now();

    Model {
        _stream_handle: stream_handle,
        sink,
        _output_stream: output_stream,
        audio_data,
        current_song,
        is_paused,
        song_history,
        history_position,
        show_song_picker,
        all_songs,
        selected_song_index,
        song_picker_scroll,
        song_hover_index,
        last_click_time,
        audio_receiver,
        visualizer,
        star_field,
        merkaba,
        show_fps: true,
        hide_ui: false,
        last_fps_update: Instant::now(),
        frame_count: 0,
        fps: 0.0,
        mouse_position: Point2::new(0.0, 0.0),
        mouse_influence: 0.0,
        fft_planner: FftPlanner::new(),
        window_function,
        volume: settings.volume,  // Use loaded volume
        volume_changed_time: Instant::now(),
        show_volume_indicator: false,
    }
}

// Update function to process real audio data and mouse input
fn update(app: &App, model: &mut Model, update: Update) {
    // Update FPS counter
    model.frame_count += 1;
    let now = Instant::now();
    let elapsed = now.duration_since(model.last_fps_update).as_secs_f32();

    if elapsed >= 1.0 {
        model.fps = model.frame_count as f32 / elapsed;
        model.frame_count = 0;
        model.last_fps_update = now;
    }

    // Get mouse position in model coordinates
    let mouse_pos = app.mouse.position();
    model.mouse_position = mouse_pos;

    // Calculate mouse distance from center
    let distance_from_center = (mouse_pos.x.powi(2) + mouse_pos.y.powi(2)).sqrt();

    // Define a threshold distance for mouse influence
    let influence_threshold = 900.0;  // Increased from 300.0 to 900.0 (3x)

    // Adjust mouse influence based on distance
    if distance_from_center < influence_threshold {
        // Smoothly increase influence as mouse gets closer to center
        model.mouse_influence += (1.0 - model.mouse_influence) * 5.0 * update.since_last.as_secs_f32();
    } else {
        // Smoothly decrease influence as mouse moves away
        model.mouse_influence += (0.0 - model.mouse_influence) * 3.0 * update.since_last.as_secs_f32();
    }

    // Clamp mouse influence to valid range
    model.mouse_influence = model.mouse_influence.clamp(0.0, 1.0);

    // Check if we need to play a new song
    if model.sink.empty() {
        next_song(model);
    }

    // Check for new audio data
    let is_paused = model.is_paused.load(Ordering::Relaxed);

    if !is_paused {
        // Try to receive audio data without blocking
        if let Ok(audio_chunk) = model.audio_receiver.try_recv() {
            // Process the audio data using FFT
            process_audio_data(model, audio_chunk);
        }
    } else {
        // If paused, gradually reduce spectrum values
        if let Ok(mut data) = model.audio_data.lock() {
            let mut spectrum = data.spectrum.clone();
            for i in 0..spectrum.len() {
                spectrum[i] *= 0.95; // Decay factor
            }

            // Generate minimal raw samples (very quiet)
            let raw_samples = vec![0.0; 1024];

            // Update with decaying spectrum
            data.update(spectrum, raw_samples, 0.0);
        }
    }

    // Get a copy of audio data for visualization
    let audio_data = if let Ok(data) = model.audio_data.lock() {
        data.clone()
    } else {
        return; // Skip this frame if we can't access audio data
    };

    // Update visualizer with delta time and mouse data
    model.visualizer.update(
        update.since_last.as_secs_f32(),
        &audio_data,
        model.mouse_position,
        model.mouse_influence
    );

    // Update star field
    model.star_field.update(update.since_last.as_secs_f32(), &audio_data);

    // Collect some reflection sample points from visualizer and stars
    let mut reflection_points = Vec::new();

    // Add some star positions for reflection
    for star in &model.star_field.stars {
        // Calculate perspective projection for the star
        let z_factor = (STAR_FIELD_DEPTH - star.z) / STAR_FIELD_DEPTH;
        let projected_x = star.x * z_factor;
        let projected_y = star.y * z_factor;

        reflection_points.push(Point2::new(projected_x, projected_y));
    }

    // Add visualization points for reflection
    for i in 0..NUM_BANDS {
        if !model.visualizer.history[i].is_empty() {
            let (x, y) = *model.visualizer.history[i].last().unwrap();
            reflection_points.push(Point2::new(x, y));
        }
    }

    // Update merkaba - pass mouse position
    model.merkaba.update(
        update.since_last.as_secs_f32(),
        &audio_data,
        reflection_points,
        model.mouse_position
    );

    // Check if volume indicator should be hidden
    if model.show_volume_indicator {
        let elapsed = Instant::now().duration_since(model.volume_changed_time).as_secs_f32();
        if elapsed > VOLUME_INDICATOR_FADE_TIME {
            model.show_volume_indicator = false;
        }
    }
}

// Updated key_pressed function with song picker controls
fn key_pressed(_app: &App, model: &mut Model, key: Key) {
    // If song picker is shown, handle its navigation keys
    if model.show_song_picker {
        match key {
            Key::S => {
                // Toggle song picker off
                model.show_song_picker = false;
            },
            Key::Up => {
                // Navigate up in song list
                if model.selected_song_index > 0 {
                    model.selected_song_index -= 1;

                    // Adjust scroll if needed
                    if model.selected_song_index < model.song_picker_scroll {
                        model.song_picker_scroll = model.selected_song_index;
                    }
                }
            },
            Key::Down => {
                // Navigate down in song list
                if model.selected_song_index < model.all_songs.len() - 1 {
                    model.selected_song_index += 1;

                    // Adjust scroll if needed
                    let max_visible_items = 30;  // This should be calculated like in the view function
                    if model.selected_song_index >= model.song_picker_scroll + max_visible_items {
                        model.song_picker_scroll = model.selected_song_index - max_visible_items + 1;
                    }
                }
            },
            Key::Return => {
                // Play selected song
                if !model.all_songs.is_empty() && model.selected_song_index < model.all_songs.len() {
                    let selected_path = model.all_songs[model.selected_song_index].path.clone();

                    // Add to history and update position
                    if model.history_position < model.song_history.len() {
                        model.song_history.truncate(model.history_position + 1);
                    }
                    model.song_history.push(selected_path.clone());
                    model.history_position = model.song_history.len() - 1;

                    // Play the selected song
                    play_selected_song(model, &selected_path);

                    // Hide song picker after selection
                    model.show_song_picker = false;
                }
            },
            Key::Escape => {
                // Hide song picker without selecting
                model.show_song_picker = false;
            },
            Key::H => {
                // Toggle hide UI - also hides song picker
                model.hide_ui = !model.hide_ui;
                if model.hide_ui {
                    model.show_song_picker = false;
                }
            },
            _ => {}
        }
    } else {
        // Normal key handling when song picker is not shown
        match key {
            Key::F => {
                // Toggle FPS display only if UI is visible
                if !model.hide_ui {
                    model.show_fps = !model.show_fps;
                }
            },
            Key::Period => {
                // Skip to next song (was previously Key::N or Key::Right)
                next_song(model);
            },
            Key::Comma => {
                // Go to previous song (was previously Key::Left)
                previous_song(model);
            },
            Key::Up => {
                // Increase volume
                model.volume = (model.volume + VOLUME_CHANGE_STEP).min(1.0);
                model.sink.set_volume(model.volume);
                model.volume_changed_time = Instant::now();
                model.show_volume_indicator = true;
                save_settings(model.volume);  // Save settings when volume changes
            },
            Key::Down => {
                // Decrease volume
                model.volume = (model.volume - VOLUME_CHANGE_STEP).max(0.0);
                model.sink.set_volume(model.volume);
                model.volume_changed_time = Instant::now();
                model.show_volume_indicator = true;
                save_settings(model.volume);  // Save settings when volume changes
            },
            Key::S => {
                // Show song picker only if UI is visible
                if !model.hide_ui {
                    model.show_song_picker = true;

                    // Refresh song list
                    model.all_songs = get_all_music_files();

                    // Find current song in the list to select it
                    if let Some(current_path) = model.song_history.get(model.history_position) {
                        for (i, song) in model.all_songs.iter().enumerate() {
                            if &song.path == current_path {
                                model.selected_song_index = i;

                                // Adjust scroll to make selected song visible
                                if i > 7 {
                                    model.song_picker_scroll = i - 7;
                                } else {
                                    model.song_picker_scroll = 0;
                                }
                                break;
                            }
                        }
                    }
                }
            },
            Key::Space => {
                // Pause/play
                if model.sink.is_paused() {
                    model.sink.play();
                    model.is_paused.store(false, Ordering::Relaxed);
                } else {
                    model.sink.pause();
                    model.is_paused.store(true, Ordering::Relaxed);
                }
            },
            Key::Escape => {
                save_settings(model.volume);  // Save settings before exit
                std::process::exit(0);
            },
            Key::H => {
                // Toggle hide UI
                model.hide_ui = !model.hide_ui;
            },
            _ => {}
        }
    }
}

fn view(app: &App, model: &Model, frame: Frame) {
    // Begin drawing
    let draw = app.draw();

    // Clear frame with black background
    draw.background().color(rgba(0.01, 0.01, 0.02, 1.0));

    // Get a copy of audio data for visualization
    if let Ok(audio_data) = model.audio_data.lock() {
        // Draw star field first (background layer)
        model.star_field.draw(&draw, &audio_data);

        // Draw visualization
        model.visualizer.draw(&draw, &audio_data);

        // Draw merkaba in the center
        model.merkaba.draw(&draw, &audio_data);

        // Draw volume indicator if active
        if model.show_volume_indicator {
            // Calculate alpha based on how long ago volume was changed
            let elapsed = Instant::now().duration_since(model.volume_changed_time).as_secs_f32();
            let alpha = if elapsed < VOLUME_INDICATOR_FADE_TIME {
                // Fade out over the last second
                if elapsed > VOLUME_INDICATOR_FADE_TIME - 1.0 {
                    1.0 - (elapsed - (VOLUME_INDICATOR_FADE_TIME - 1.0))
                } else {
                    1.0
                }
            } else {
                0.0
            };

            // Calculate volume bar height based on current volume
            let window_height = WINDOW_HEIGHT as f32;
            let bar_height = window_height * model.volume;
            let bar_x = (WINDOW_WIDTH as f32) / 2.0 - VOLUME_BAR_MARGIN;
            let bar_y = -window_height / 2.0 + bar_height / 2.0;

            // Draw volume bar background
            draw.rect()
                .x_y(bar_x, 0.0)
                .w_h(VOLUME_BAR_WIDTH, window_height)
                .color(rgba(0.2, 0.2, 0.2, 0.3 * alpha));

            // Draw volume level
            draw.rect()
                .x_y(bar_x, bar_y)
                .w_h(VOLUME_BAR_WIDTH, bar_height)
                .color(rgba(0.8, 0.8, 1.0, alpha));
        }

        // Only draw UI elements if not hidden
        if !model.hide_ui {
            // Draw FPS counter if enabled
            if model.show_fps {
                let fps_text = format!("{:.1} FPS", model.fps);
                // Position in top right corner
                draw.text(&fps_text)
                    .font_size(14)
                    .x_y(WINDOW_WIDTH as f32 / 2.0 - 50.0, (WINDOW_HEIGHT as f32 / 2.0) - 20.0)
                    .color(WHITE);
            }

            // Create a unified info panel at the bottom
            let bottom_y = -(WINDOW_HEIGHT as f32 / 2.0) + 30.0;

            // Display song info and controls in a uniform style
            let song_info = if !model.current_song.is_empty() {
                format!("♫ {} ♫", model.current_song)
            } else {
                "No song playing".to_string()
            };

            // Draw a semi-transparent background for text
            draw.rect()
                .x_y(0.0, bottom_y)
                .w_h(WINDOW_WIDTH as f32, 50.0)
                .color(rgba(0.0, 0.0, 0.0, 0.5));

            // Draw song info
            draw.text(&song_info)
                .font_size(14)
                .x_y(0.0, bottom_y - 8.0)
                .color(rgba(1.0, 1.0, 1.0, 0.9))
                .w(WINDOW_WIDTH as f32 * 0.9);

            // Draw controls - Updated to show new key bindings
            draw.text("[Space] Pause/Play | [,/.] Prev/Next Song | [↑/↓] Volume | [S] Song List | [F] Toggle FPS | [H] Hide UI | [Esc] Exit")
                .font_size(14)
                .x_y(0.0, bottom_y + 12.0)
                .color(rgba(1.0, 1.0, 1.0, 0.8))
                .w(WINDOW_WIDTH as f32 * 0.9);

            // Draw song picker if visible
            if model.show_song_picker {
                // Calculate dimensions
                let panel_width = WINDOW_WIDTH as f32 * 0.3;
                let panel_height = WINDOW_HEIGHT as f32;
                let panel_x = -(WINDOW_WIDTH as f32) / 2.0 + panel_width / 2.0;
                let panel_y = 0.0;

                // Draw panel background
                draw.rect()
                    .x_y(panel_x, panel_y)
                    .w_h(panel_width, panel_height)
                    .color(rgba(0.05, 0.05, 0.1, 0.9));

                // Draw header
                draw.rect()
                    .x_y(panel_x, panel_height / 2.0 - 30.0)
                    .w_h(panel_width, 60.0)
                    .color(rgba(0.1, 0.1, 0.2, 1.0));

                draw.text("Song List")
                    .font_size(24)
                    .x_y(panel_x, panel_height / 2.0 - 20.0)
                    .color(rgba(1.0, 1.0, 1.0, 0.9))
                    .w(panel_width * 0.9);

                // Update instructions to include mouse controls
                draw.text("[↑/↓] Navigate | Mouse Wheel to Scroll | Click to Play")
                    .font_size(12)
                    .x_y(panel_x, panel_height / 2.0 - 40.0)
                    .color(rgba(0.8, 0.8, 0.8, 0.8))
                    .w(panel_width * 0.9);

                // Add close button in the top-right corner
                let close_button_x = panel_x + panel_width / 2.0 - 20.0;
                let close_button_y = panel_height / 2.0 - 20.0;

                // Draw X for close button
                draw.line()
                    .start(pt2(close_button_x - 8.0, close_button_y - 8.0))
                    .end(pt2(close_button_x + 8.0, close_button_y + 8.0))
                    .weight(2.0)
                    .color(rgba(0.8, 0.8, 0.8, 0.9));

                draw.line()
                    .start(pt2(close_button_x + 8.0, close_button_y - 8.0))
                    .end(pt2(close_button_x - 8.0, close_button_y + 8.0))
                    .weight(2.0)
                    .color(rgba(0.8, 0.8, 0.8, 0.9));

                // Draw song list
                let list_start_y = panel_height / 2.0 - 80.0;
                let item_height = 28.0;  // Slightly reduced from 30.0 to fit more items

                // Calculate how many items can fit in the available space
                let bottom_panel_height = 60.0;  // Height of the bottom info panel
                let available_height = panel_height - 80.0 - bottom_panel_height;
                let max_visible_items = (available_height / item_height).floor() as usize;

                // Get the visible range of songs
                let end_idx = (model.song_picker_scroll + max_visible_items).min(model.all_songs.len());
                let visible_songs = &model.all_songs[model.song_picker_scroll..end_idx];

                // Draw each visible song
                for (i, song) in visible_songs.iter().enumerate() {
                    let y_pos = list_start_y - (i as f32 * item_height);
                    let absolute_index = model.song_picker_scroll + i;
                    let is_selected = absolute_index == model.selected_song_index;
                    let is_hovered = model.song_hover_index == Some(absolute_index);

                    // Draw selection/hover highlight with different colors
                    if is_selected && is_hovered {
                        // Both selected and hovered - brightest
                        draw.rect()
                            .x_y(panel_x, y_pos)
                            .w_h(panel_width, item_height)
                            .color(rgba(0.3, 0.4, 0.6, 0.8));
                    } else if is_selected {
                        // Selected only
                        draw.rect()
                            .x_y(panel_x, y_pos)
                            .w_h(panel_width, item_height)
                            .color(rgba(0.2, 0.3, 0.5, 0.7));
                    } else if is_hovered {
                        // Hovered only
                        draw.rect()
                            .x_y(panel_x, y_pos)
                            .w_h(panel_width, item_height)
                            .color(rgba(0.15, 0.25, 0.4, 0.6));
                    }

                    // Draw song name with metadata
                    draw.text(&song.display_name)
                        .font_size(14)
                        .x_y(panel_x, y_pos)
                        .color(if is_selected || is_hovered { rgba(1.0, 1.0, 1.0, 1.0) } else { rgba(0.8, 0.8, 0.8, 0.8) })
                        .w(panel_width * 0.9);
                }

                // Draw scroll indicators if needed
                if model.song_picker_scroll > 0 {
                    // Draw an up arrow
                    let arrow_x = panel_x;
                    let arrow_y = panel_height / 2.0 - 90.0;
                    draw.polygon()
                        .points([
                            pt2(arrow_x, arrow_y + 5.0),
                            pt2(arrow_x - 10.0, arrow_y - 5.0),
                            pt2(arrow_x + 10.0, arrow_y - 5.0),
                        ])
                        .color(rgba(1.0, 1.0, 1.0, 0.6));
                }

                if end_idx < model.all_songs.len() {
                    // Draw a down arrow
                    let arrow_x = panel_x;
                    let arrow_y = list_start_y - (max_visible_items as f32 * item_height);
                    draw.polygon()
                        .points([
                            pt2(arrow_x, arrow_y - 5.0),
                            pt2(arrow_x - 10.0, arrow_y + 5.0),
                            pt2(arrow_x + 10.0, arrow_y + 5.0),
                        ])
                        .color(rgba(1.0, 1.0, 1.0, 0.6));
                }

                // Draw a scrollbar
                if model.all_songs.len() > max_visible_items {
                    let scrollbar_width = 8.0;
                    let scrollbar_x = panel_x + (panel_width / 2.0) - (scrollbar_width / 2.0) - 10.0;
                    let scrollbar_height = available_height;
                    let scrollbar_y = list_start_y - (scrollbar_height / 2.0);

                    // Draw scrollbar background
                    draw.rect()
                        .x_y(scrollbar_x, scrollbar_y)
                        .w_h(scrollbar_width, scrollbar_height)
                        .color(rgba(0.1, 0.1, 0.15, 0.4));

                    // Draw scrollbar handle
                    let handle_size_ratio = max_visible_items as f32 / model.all_songs.len() as f32;
                    let handle_height = scrollbar_height * handle_size_ratio;
                    let handle_position_ratio = model.song_picker_scroll as f32 /
                                               (model.all_songs.len() - max_visible_items).max(1) as f32;
                    let handle_y_offset = (scrollbar_height - handle_height) * handle_position_ratio;
                    let handle_y = scrollbar_y + (scrollbar_height / 2.0) - handle_height / 2.0 - handle_y_offset;

                    draw.rect()
                        .x_y(scrollbar_x, handle_y)
                        .w_h(scrollbar_width, handle_height)
                        .color(rgba(0.5, 0.5, 0.6, 0.8));
                }
            }
        }
    }

    // Finalize drawing
    draw.to_frame(app, &frame).unwrap();
}

// Clone implementation for AudioData
impl Clone for AudioData {
    fn clone(&self) -> Self {
        Self {
            spectrum: self.spectrum.clone(),
            raw_samples: self.raw_samples.clone(),
            power: self.power,
        }
    }
}

// New function to handle mouse wheel events for scrolling the song list
fn mouse_wheel(_app: &App, model: &mut Model, delta: MouseScrollDelta, _phase: TouchPhase) {
    if model.show_song_picker && !model.all_songs.is_empty() {
        // Convert delta to a scroll amount
        let scroll_amount = match delta {
            MouseScrollDelta::LineDelta(_x, y) => {
                if y > 0.0 { -3 } else if y < 0.0 { 3 } else { 0 }
            },
            MouseScrollDelta::PixelDelta(delta) => {
                if delta.y > 0.0 { -3 } else if delta.y < 0.0 { 3 } else { 0 }
            }
        };

        // Apply scroll keeping within bounds
        if scroll_amount > 0 {
            // Scroll down
            let max_scroll = model.all_songs.len().saturating_sub(1);
            model.song_picker_scroll = (model.song_picker_scroll + scroll_amount as usize).min(max_scroll);
        } else if scroll_amount < 0 && model.song_picker_scroll > 0 {
            // Scroll up
            model.song_picker_scroll = model.song_picker_scroll.saturating_sub((-scroll_amount) as usize);
        }
    }
}

// New function to handle mouse movement for hover effects
fn mouse_moved(app: &App, model: &mut Model, _position: Point2) {
    if model.show_song_picker {
        // Get mouse position
        let mouse_pos = app.mouse.position();

        // Calculate song list dimensions and position
        let panel_width = WINDOW_WIDTH as f32 * 0.3;
        let panel_x = -(WINDOW_WIDTH as f32) / 2.0 + panel_width / 2.0;
        let panel_left = panel_x - panel_width / 2.0;
        let panel_right = panel_x + panel_width / 2.0;

        let list_start_y = (WINDOW_HEIGHT as f32) / 2.0 - 80.0;
        let item_height = 28.0;

        // Check if mouse is inside the song list area
        if mouse_pos.x >= panel_left && mouse_pos.x <= panel_right {
            // Calculate which song the mouse is hovering over
            let relative_y = list_start_y - mouse_pos.y;

            if relative_y >= 0.0 {
                let index = (relative_y / item_height).floor() as usize;

                // Make sure index is within the visible range
                let max_visible_items = ((WINDOW_HEIGHT as f32 - 140.0) / item_height).floor() as usize;
                let end_idx = (model.song_picker_scroll + max_visible_items).min(model.all_songs.len());

                if index < end_idx - model.song_picker_scroll {
                    model.song_hover_index = Some(model.song_picker_scroll + index);
                } else {
                    model.song_hover_index = None;
                }
            } else {
                model.song_hover_index = None;
            }
        } else {
            model.song_hover_index = None;
        }
    }
}

// New function to handle mouse clicks for song selection
fn mouse_pressed(app: &App, model: &mut Model, button: MouseButton) {
    if model.show_song_picker && button == MouseButton::Left {
        if let Some(hover_index) = model.song_hover_index {
            if hover_index < model.all_songs.len() {
                // Select the hovered song
                model.selected_song_index = hover_index;

                // Play the selected song immediately with a single click
                let selected_path = model.all_songs[model.selected_song_index].path.clone();

                // Add to history and update position
                if model.history_position < model.song_history.len() {
                    model.song_history.truncate(model.history_position + 1);
                }
                model.song_history.push(selected_path.clone());
                model.history_position = model.song_history.len() - 1;

                // Play the selected song
                play_selected_song(model, &selected_path);

                // Hide song picker after selection
                model.show_song_picker = false;
            }
        }

        // Check if the user clicked on the close button (top-right of panel)
        let panel_width = WINDOW_WIDTH as f32 * 0.3;
        let panel_x = -(WINDOW_WIDTH as f32) / 2.0 + panel_width / 2.0;
        let panel_right = panel_x + panel_width / 2.0;
        let panel_top = (WINDOW_HEIGHT as f32) / 2.0;

        let close_button_x = panel_right - 20.0;
        let close_button_y = panel_top - 20.0;
        let mouse_pos = app.mouse.position();

        let distance = ((mouse_pos.x - close_button_x).powi(2) +
                        (mouse_pos.y - close_button_y).powi(2)).sqrt();

        if distance < 15.0 {
            // Close the song picker
            model.show_song_picker = false;
        }
    }
}

// Function to save settings to a file
fn save_settings(volume: f32) {
    let settings = Settings { volume };
    if let Ok(serialized) = serde_json::to_string_pretty(&settings) {
        if let Ok(mut file) = std::fs::File::create(SETTINGS_FILE) {
            let _ = file.write_all(serialized.as_bytes());
        }
    }
}

// Function to load settings from a file
fn load_settings() -> Settings {
    match std::fs::File::open(SETTINGS_FILE) {
        Ok(file) => {
            match serde_json::from_reader(file) {
                Ok(settings) => settings,
                Err(_) => Settings::default(),
            }
        },
        Err(_) => Settings::default(),
    }
}

// Settings structure for persistence
#[derive(serde::Serialize, serde::Deserialize)]
struct Settings {
    volume: f32,
}

impl Settings {
    fn default() -> Self {
        Self {
            volume: DEFAULT_VOLUME,
        }
    }
}