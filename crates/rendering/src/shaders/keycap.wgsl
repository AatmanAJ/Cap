const MAX_KEYCAPS: u32 = 8u;

struct Keycap {
    rect: vec4<f32>,
    params: vec4<f32>,
};

struct KeycapUniforms {
    face_color: vec4<f32>,
    edge_color: vec4<f32>,
    globals: vec4<f32>,
    keycaps: array<Keycap, MAX_KEYCAPS>,
};

@group(0) @binding(0) var<uniform> uniforms: KeycapUniforms;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var positions = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(1.0, 1.0),
    );

    var output: VertexOutput;
    output.position = vec4<f32>(positions[vertex_index], 0.0, 1.0);
    return output;
}

fn rounded_rect_sdf(p: vec2<f32>, rect: vec4<f32>, radius: f32) -> f32 {
    let half_size = rect.zw * 0.5;
    let center = rect.xy + half_size;
    let r = min(radius, min(half_size.x, half_size.y));
    let q = abs(p - center) - half_size + vec2<f32>(r);
    return length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0) - r;
}

fn coverage(distance: f32) -> f32 {
    return clamp(0.5 - distance, 0.0, 1.0);
}

fn over(dst: vec4<f32>, rgb: vec3<f32>, alpha: f32) -> vec4<f32> {
    return vec4<f32>(rgb * alpha + dst.rgb * (1.0 - alpha), alpha + dst.a * (1.0 - alpha));
}

@fragment
fn fs_main(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let p = position.xy;
    let opacity = uniforms.globals.x;
    let shadow_alpha = uniforms.globals.y;
    let shadow_blur = max(uniforms.globals.z, 0.5);
    let count = min(u32(uniforms.globals.w), MAX_KEYCAPS);

    var color = vec4<f32>(0.0);

    for (var i = 0u; i < count; i++) {
        let keycap = uniforms.keycaps[i];
        let depth = keycap.params.y;
        let radius = keycap.params.z;
        let shadow_rect = keycap.rect + vec4<f32>(0.0, depth + shadow_blur * 0.35, 0.0, 0.0);
        let distance = rounded_rect_sdf(p, shadow_rect, radius);
        let shadow = shadow_alpha * (1.0 - smoothstep(-shadow_blur, shadow_blur, distance));
        color = over(color, vec3<f32>(0.0), shadow);
    }

    for (var i = 0u; i < count; i++) {
        let keycap = uniforms.keycaps[i];
        let press = keycap.params.x;
        let depth = keycap.params.y;
        let radius = keycap.params.z;

        let skirt_rect = keycap.rect + vec4<f32>(0.0, depth, 0.0, 0.0);
        color = over(
            color,
            uniforms.edge_color.rgb,
            uniforms.edge_color.a * coverage(rounded_rect_sdf(p, skirt_rect, radius)),
        );

        let face_rect = keycap.rect + vec4<f32>(0.0, press, 0.0, 0.0);
        let face_distance = rounded_rect_sdf(p, face_rect, radius);
        let along = clamp((p.y - face_rect.y) / max(face_rect.w, 1.0), 0.0, 1.0);
        let sheen = 0.10 * (1.0 - smoothstep(0.0, 0.6, along));
        let rim = (1.0 - smoothstep(0.0, 1.5, -face_distance)) * (1.0 - smoothstep(0.0, 0.35, along));
        let face = mix(uniforms.face_color.rgb, vec3<f32>(1.0), clamp(sheen + rim * 0.22, 0.0, 1.0));
        color = over(color, face, uniforms.face_color.a * coverage(face_distance));
    }

    if color.a <= 0.0 {
        return vec4<f32>(0.0);
    }
    return vec4<f32>(color.rgb / color.a, color.a * opacity);
}
