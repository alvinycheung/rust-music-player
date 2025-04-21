// Vertex shader

struct VertexInput {
    @location(0) position: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
};

@vertex
fn vs_main(
    input: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = vec4<f32>(input.position, 0.0, 1.0);
    out.tex_coords = input.position * 0.5 + 0.5; // Convert from clip space to texture coordinates
    return out;
}

// Fragment shader

struct Uniforms {
    center: vec2<f32>,
    zoom: f32,
    aspect_ratio: f32,
    iterations: u32,
    color_offset: f32,
    julia_c: vec2<f32>,
    pad: vec2<f32>, // Padding for alignment
}

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

// HSV to RGB conversion
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> vec3<f32> {
    let c = v * s;
    let h_prime = h / 60.0;
    let x = c * (1.0 - abs(fract(h_prime / 2.0) * 2.0 - 1.0));
    let m = v - c;

    var rgb: vec3<f32>;

    if (h_prime < 1.0) {
        rgb = vec3<f32>(c, x, 0.0);
    } else if (h_prime < 2.0) {
        rgb = vec3<f32>(x, c, 0.0);
    } else if (h_prime < 3.0) {
        rgb = vec3<f32>(0.0, c, x);
    } else if (h_prime < 4.0) {
        rgb = vec3<f32>(0.0, x, c);
    } else if (h_prime < 5.0) {
        rgb = vec3<f32>(x, 0.0, c);
    } else {
        rgb = vec3<f32>(c, 0.0, x);
    }

    return rgb + vec3<f32>(m);
}

// Julia set calculation with smooth coloring
fn julia_set(z: vec2<f32>, c: vec2<f32>, max_iter: u32) -> f32 {
    var zr: f32 = z.x;
    var zi: f32 = z.y;
    var iter: u32 = 0u;

    // We'll track these for smooth coloring
    var escaped: bool = false;
    var escape_value: f32 = 0.0;
    var last_zr: f32 = 0.0;
    var last_zi: f32 = 0.0;

    // Julia set iteration
    while (iter < max_iter) {
        // z = z² + c
        let zr_temp = zr * zr - zi * zi + c.x;
        zi = 2.0 * zr * zi + c.y;
        zr = zr_temp;

        let mag_squared = zr * zr + zi * zi;

        // Check if we've escaped
        if (mag_squared > 4.0) {
            escaped = true;

            // Calculate smooth iteration count for better coloring
            // Using a logarithmic smooth coloring algorithm
            let log_zn = log(mag_squared) / 2.0;
            let nu = log(log_zn / log(2.0)) / log(2.0);
            escape_value = f32(iter) - nu;

            break;
        }

        last_zr = zr;
        last_zi = zi;
        iter = iter + 1u;
    }

    if (escaped) {
        return escape_value / f32(max_iter);
    } else {
        return 1.0; // Inside the set
    }
}

// Fragment shader main function
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Convert screen coordinates to complex plane
    let zoom_factor = exp(uniforms.zoom);
    let scale = 1.5 / zoom_factor;

    // Apply aspect ratio correction
    let aspect_corrected_x = (in.tex_coords.x - 0.5) * uniforms.aspect_ratio;

    // Convert to complex coordinates with zoom
    let z = vec2<f32>(
        aspect_corrected_x * scale + uniforms.center.x,
        (in.tex_coords.y - 0.5) * scale + uniforms.center.y
    );

    // Calculate Julia set
    let value = julia_set(z, uniforms.julia_c, uniforms.iterations);

    // Color mapping
    if (value >= 1.0) {
        // Inside the set - black
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    } else {
        // Outside the set with smooth coloring

        // Calculate hue based on value and add animation offset
        let hue = fract(value * 8.0 + uniforms.color_offset / 360.0) * 360.0;

        // Brightness based on distance from the set (inverse)
        let brightness = 0.7 + 0.3 * (1.0 - value);

        // Convert HSV to RGB
        let color = hsv_to_rgb(hue, 0.9, brightness);

        return vec4<f32>(color, 1.0);
    }
}