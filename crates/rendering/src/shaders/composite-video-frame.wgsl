struct Uniforms {
    crop_bounds: vec4<f32>,
    target_bounds: vec4<f32>,
    output_size: vec2<f32>,
    frame_size: vec2<f32>,
    motion_blur_vector: vec2<f32>,
    motion_blur_zoom_center: vec2<f32>,
    motion_blur_params: vec4<f32>,
    target_size: vec2<f32>,
    rounding_px: f32,
    rounding_type: f32,
    mirror_x: f32,
    shadow: f32,
    shadow_size: f32,
    shadow_opacity: f32,
    shadow_blur: f32,
    opacity: f32,
    border_enabled: f32,
    border_width: f32,
    preserve_source_alpha: f32,
    _padding1a: f32,
    _padding1b: f32,
    _padding1c: f32,
    border_color: vec4<f32>,
    // Per-corner multipliers on rounding_px: (tl, tr, bl, br). All 1s keeps
    // the uniform rounding; the display squares its top corners against
    // decorative frame chrome with (0, 0, 1, 1).
    corner_radii: vec4<f32>,
    // (exposure stops, contrast, saturation, temperature).
    color_adjust_a: vec4<f32>,
    // (tint, fade, split_tone, vignette).
    color_adjust_b: vec4<f32>,
    // (grain amount, grain seed, grade-active flag, full-frame vignette flag).
    grain_params: vec4<f32>,
    loupe_position: vec4<f32>,
    loupe_shape: vec4<f32>,
    loupe_motion: vec4<f32>,
    sampling_params: vec4<f32>,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var frame_texture: texture_2d<f32>;
@group(0) @binding(2) var frame_sampler: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0)
    );

    var out: VertexOutput;
    out.position = vec4<f32>(positions[vertex_index], 0.0, 1.0);
    out.uv = (positions[vertex_index] + 1.0) * 0.5;
    return out;
}

// Safety ceiling on the radial zoom-blur ray length in card-UV. Real zoom
// springs produce rays around 0.03-0.05 at the far corners; this only guards
// synthetic or pathological amounts.
const MAX_ZOOM_RAY_UV: f32 = 0.10;

// Kept mild and clamped to the local min/max so sharpening can't ring.
const SHARPEN_MAX_LOBE: f32 = 0.2;
const SHARPEN_FULL_UPSCALE: f32 = 2.5;

const LOUPE_REFRACTION: f32 = 0.22;
const LOUPE_CHROMA: f32 = 0.012;
const LOUPE_SHADOW_OPACITY: f32 = 0.38;
const LOUPE_BLUR_TAPS: i32 = 12;

// Per-pixel random phase used to dither the fixed 13-tap zoom kernel
// (deterministic in the fragment position, so renders stay reproducible).
fn interleaved_noise(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(12.9898, 78.233))) * 43758.5453);
}

fn superellipse_norm(p: vec2<f32>, power: f32) -> f32 {
    let x = pow(abs(p.x), power);
    let y = pow(abs(p.y), power);
    return pow(x + y, 1.0 / power);
}

fn rounded_corner_norm(p: vec2<f32>, rounding_type: f32) -> f32 {
    if rounding_type < 0.5 {
        return length(p);
    }

    let power = 4.0;
    return superellipse_norm(p, power);
}

// Radius for the corner of the quadrant `p` (center-relative) falls in.
fn corner_radius_for(p: vec2<f32>) -> f32 {
    let multiplier = select(
        select(uniforms.corner_radii.w, uniforms.corner_radii.z, p.x < 0.0),
        select(uniforms.corner_radii.y, uniforms.corner_radii.x, p.x < 0.0),
        p.y < 0.0
    );
    return uniforms.rounding_px * multiplier;
}

fn sdf_rounded_rect(p: vec2<f32>, b: vec2<f32>, r: f32, rounding_type: f32) -> f32 {
    let q = abs(p) - b + vec2<f32>(r);
    let outside = max(q, vec2<f32>(0.0));
    let outside_norm = rounded_corner_norm(outside, rounding_type);
    return outside_norm + min(max(q.x, q.y), 0.0) - r;
}

fn coverage_from_distance(distance: f32, anti_alias_width: f32) -> f32 {
    return clamp(1.0 - smoothstep(-anti_alias_width, anti_alias_width, distance), 0.0, 1.0);
}

fn rounded_rect_coverage(p: vec2<f32>, b: vec2<f32>, r: f32, rounding_type: f32) -> f32 {
    let distance = sdf_rounded_rect(p, b, r, rounding_type);
    let anti_alias_width = max(fwidth(distance), 1.0);

    if distance <= -anti_alias_width {
        return 1.0;
    }

    if distance >= anti_alias_width {
        return 0.0;
    }

    let subpixel_offset = 0.25;
    var coverage = 0.0;
    coverage += coverage_from_distance(
        sdf_rounded_rect(p + vec2<f32>(-subpixel_offset, -subpixel_offset), b, r, rounding_type),
        anti_alias_width
    );
    coverage += coverage_from_distance(
        sdf_rounded_rect(p + vec2<f32>(subpixel_offset, -subpixel_offset), b, r, rounding_type),
        anti_alias_width
    );
    coverage += coverage_from_distance(
        sdf_rounded_rect(p + vec2<f32>(-subpixel_offset, subpixel_offset), b, r, rounding_type),
        anti_alias_width
    );
    coverage += coverage_from_distance(
        sdf_rounded_rect(p + vec2<f32>(subpixel_offset, subpixel_offset), b, r, rounding_type),
        anti_alias_width
    );
    return coverage * 0.25;
}

// Sin-free hash so grain stays stable across GPU drivers.
fn grain_hash(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3<f32>(p.x, p.y, p.x) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

// Must stay in sync with color-grade.wgsl and cursor.wgsl. Runs once per
// output pixel after motion-blur resolve; ungraded layers skip everything in
// a single coherent uniform branch.
fn apply_color_grade(color: vec4<f32>, target_uv: vec2<f32>, frag_pos: vec2<f32>) -> vec4<f32> {
    if uniforms.grain_params.z < 0.5 {
        return color;
    }

    let exposure = uniforms.color_adjust_a.x;
    let contrast = uniforms.color_adjust_a.y;
    let saturation = uniforms.color_adjust_a.z;
    let temperature = uniforms.color_adjust_a.w;
    let tint = uniforms.color_adjust_b.x;
    let fade = uniforms.color_adjust_b.y;
    let split_tone = uniforms.color_adjust_b.z;
    let vignette = uniforms.color_adjust_b.w;
    let grain = uniforms.grain_params.x;

    // Exposure in stops.
    var rgb = color.rgb * exp2(exposure);

    // White balance, multiplicative so black stays black.
    rgb = rgb * vec3<f32>(
        1.0 + 0.10 * temperature + 0.04 * tint,
        1.0 - 0.07 * tint,
        1.0 - 0.10 * temperature + 0.04 * tint,
    );

    // Contrast around mid gray.
    rgb = (rgb - vec3<f32>(0.5)) * (1.0 + contrast) + vec3<f32>(0.5);

    // Saturation via luma mix (-1 = grayscale).
    let luma = dot(clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(0.2126, 0.7152, 0.0722));
    rgb = mix(vec3<f32>(luma), rgb, 1.0 + saturation);

    // Split tone: teal shadows / orange highlights, luma-banded.
    let shadow_w = 1.0 - smoothstep(0.2, 0.65, luma);
    let highlight_w = smoothstep(0.35, 0.8, luma);
    rgb += split_tone * (
        shadow_w * vec3<f32>(-0.06, 0.02, 0.08) +
        highlight_w * vec3<f32>(0.08, 0.02, -0.06)
    );

    // Film fade: raised blacks, gently dulled highlights.
    rgb = rgb * (1.0 - 0.18 * fade) + vec3<f32>(0.09 * fade);

    if vignette > 0.0 {
        var vig_uv = target_uv;
        if uniforms.grain_params.w > 0.5 {
            vig_uv = frag_pos / uniforms.output_size;
        }
        let r = length((vig_uv - vec2<f32>(0.5)) * 2.0);
        rgb = rgb * (1.0 - vignette * 0.65 * smoothstep(0.5, 1.5, r));
    }

    // Midtone-peaked monochrome grain.
    if grain > 0.0 {
        let seed = uniforms.grain_params.y;
        let noise = grain_hash(frag_pos + vec2<f32>(seed * 17.0, seed * 29.0));
        let graded_luma = dot(
            clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0)),
            vec3<f32>(0.2126, 0.7152, 0.0722)
        );
        let response = 0.25 + 0.75 * (1.0 - abs(2.0 * graded_luma - 1.0));
        rgb += (noise - 0.5) * grain * 0.35 * response;
    }

    return vec4<f32>(clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0)), color.a);
}

fn composite_source_over(foreground: vec4<f32>, background: vec4<f32>) -> vec4<f32> {
    let alpha = foreground.a + background.a * (1.0 - foreground.a);

    if alpha <= 0.0001 {
        return vec4<f32>(0.0);
    }

    let color = (
        foreground.rgb * foreground.a +
        background.rgb * background.a * (1.0 - foreground.a)
    ) / alpha;

    return vec4<f32>(color, alpha);
}

@fragment
fn fs_main(@builtin(position) frag_coord: vec4<f32>) -> @location(0) vec4<f32> {
    let p = frag_coord.xy;
    let center = (uniforms.target_bounds.xy + uniforms.target_bounds.zw) * 0.5;
    let size = (uniforms.target_bounds.zw - uniforms.target_bounds.xy) * 0.5;
    
    let dist = sdf_rounded_rect(p - center, size, corner_radius_for(p - center), uniforms.rounding_type);

    let min_frame_size = min(size.x, size.y);
    let shadow_enabled = uniforms.shadow > 0.0;

    // Get shadow parameters, either from advanced settings or fallback to basic shadow
    let shadow_strength = uniforms.shadow / 100.0;

    // Use shadow_strength as a multiplier for all advanced settings
    let shadow_size = select(
        shadow_strength * min_frame_size,
        shadow_strength * (uniforms.shadow_size / 100.0) * min_frame_size,
        shadow_enabled
    );

    let shadow_opacity = select(
        shadow_strength * 0.18,
        shadow_strength * (uniforms.shadow_opacity / 100.0),
        shadow_enabled
    );

    let shadow_blur = select(
        shadow_strength * min_frame_size * 0.5,
        shadow_strength * (uniforms.shadow_blur / 100.0) * min_frame_size,
        shadow_enabled
    );

    let shadow_dist = dist;

    // Apply blur and size to shadow
    let shadow_strength_final = smoothstep(shadow_size + shadow_blur, -shadow_blur, abs(shadow_dist));
    let shadow_color = vec4<f32>(0.0, 0.0, 0.0, shadow_strength_final * shadow_opacity);

    let target_uv = (p - uniforms.target_bounds.xy) / uniforms.target_size;
    let crop_bounds_uv = vec4<f32>(uniforms.crop_bounds.xy / uniforms.frame_size, uniforms.crop_bounds.zw / uniforms.frame_size);
    let edge_padding = max(2.0, uniforms.border_width + 2.0);
    let edge_padding_uv = edge_padding / uniforms.target_size;

    // How far outside the card the motion smear can land (card-UV). Zero when
    // blur is inactive so the early-outs below stay as tight as before.
    let blur_mode = uniforms.motion_blur_params.x;
    let blur_strength = uniforms.motion_blur_params.y;
    let blur_active = blur_mode >= 0.5 && blur_strength >= 0.001;
    var blur_reach_uv = vec2<f32>(0.0);
    if blur_active {
        if blur_mode < 1.5 {
            blur_reach_uv = abs(uniforms.motion_blur_vector);
        } else {
            blur_reach_uv = vec2<f32>(MAX_ZOOM_RAY_UV);
        }
    }

    // Evaluate coverage before the apron return so fwidth retains every helper lane.
    let shape_coverage = rounded_rect_coverage(
        p - center,
        size,
        corner_radius_for(p - center),
        uniforms.rounding_type
    );
    var border_coverage = 0.0;
    if (uniforms.border_enabled > 0.0) {
        let border_outer_coverage = rounded_rect_coverage(
            p - center,
            size + vec2<f32>(uniforms.border_width),
            corner_radius_for(p - center) + uniforms.border_width,
            uniforms.rounding_type
        );
        border_coverage = clamp(border_outer_coverage - shape_coverage, 0.0, 1.0);
    }

    if target_uv.x < -edge_padding_uv.x - blur_reach_uv.x ||
        target_uv.x > 1.0 + edge_padding_uv.x + blur_reach_uv.x ||
        target_uv.y < -edge_padding_uv.y - blur_reach_uv.y ||
        target_uv.y > 1.0 + edge_padding_uv.y + blur_reach_uv.y
    {
        return shadow_color;
    }

    if (border_coverage > 0.001) {
        let border_color = vec4<f32>(
            uniforms.border_color.xyz,
            border_coverage * uniforms.border_color.w
        );
        return composite_source_over(border_color, shadow_color);
    }

    // Outside the card the blur can still land content (the smear escapes the
    // card edge along the motion), so only take the fast exit when inactive.
    if shape_coverage <= 0.001 && !blur_active {
        return shadow_color;
    }

    let sample_target_uv = clamp(target_uv, vec2<f32>(0.0), vec2<f32>(1.0));
    var base_color = sample_display(sample_target_uv, crop_bounds_uv);
    base_color.a = base_color.a * shape_coverage * uniforms.opacity;

    let zoom_amount = uniforms.motion_blur_params.z;

    if !blur_active {
        return finish_card(base_color, target_uv, p, crop_bounds_uv, shadow_color);
    }

    // Length-based semantics: the user amount is baked into the LENGTH of
    // the kernel (velocity vector / zoom ray) on the CPU side, and the output
    // is the fully blurred result — never a crossfade with the sharp frame
    // (a sharp copy mixed over a smear reads as ghosting, not motion). Alpha
    // is accumulated with the same kernel so the card edge smears along the
    // motion instead of staying crisp around streaked content. A zero-length
    // kernel is the identity, so blur fades in and out with velocity
    // continuously and needs no strength ramp.
    if blur_mode < 1.5 {
        let velocity_uv = uniforms.motion_blur_vector;
        if length(velocity_uv) < 1e-5 {
            return finish_card(base_color, target_uv, p, crop_bounds_uv, shadow_color);
        }

        // 21-tap box along [0, +v]: matches the reference directional filter
        // (kernel 21 with offset -|v|/2, which anchors the span at the pixel).
        var accum = vec3<f32>(0.0);
        var alpha_sum = 0.0;
        let k = 20.0;

        for (var i = 0; i <= 20; i = i + 1) {
            let sample_uv = target_uv + velocity_uv * (f32(i) / k);
            var tap = sample_texture(sample_uv, crop_bounds_uv);
            tap = apply_rounded_corners(tap, sample_uv);
            accum += tap.rgb * tap.a;
            alpha_sum += tap.a;
        }

        let out_alpha = (alpha_sum / 21.0) * uniforms.opacity;
        if out_alpha <= 0.0001 || alpha_sum <= 0.0001 {
            return shadow_color;
        }
        let smear_px = length(velocity_uv * uniforms.target_size);
        return finish_card(
            vec4(
                with_sharp_detail(accum / alpha_sum, base_color, sample_target_uv, crop_bounds_uv, smear_px),
                out_alpha
            ),
            target_uv,
            p,
            crop_bounds_uv,
            shadow_color
        );
    }

    let zoom_center = uniforms.motion_blur_zoom_center;
    let dir = zoom_center - target_uv;
    let center_dist = length(dir);
    if center_dist < 1e-4 || zoom_amount < 1e-4 {
        return finish_card(base_color, target_uv, p, crop_bounds_uv, shadow_color);
    }

    // Radial blur toward the scale origin: ray length grows with distance
    // from the center (ray = center_dist * amount), parabolic weights peaked
    // mid-ray, and a per-pixel random phase to dither the 13-tap banding —
    // all matching the reference zoom filter.
    let scaled_dir = dir / center_dist * min(center_dist * min(zoom_amount, 1.0), MAX_ZOOM_RAY_UV);
    let max_kernel = 13.0;
    let dither = interleaved_noise(p);

    var accum = vec3<f32>(0.0);
    var alpha_sum = 0.0;
    var weight_sum = 0.0;

    for (var i = 0; i < 13; i = i + 1) {
        let percent = (f32(i) + dither) / max_kernel;
        let weight = 4.0 * (percent - percent * percent);
        let sample_uv = target_uv + scaled_dir * percent;

        var tap = sample_texture(sample_uv, crop_bounds_uv);
        tap = apply_rounded_corners(tap, sample_uv);
        accum += tap.rgb * tap.a * weight;
        alpha_sum += tap.a * weight;
        weight_sum += weight;
    }

    if weight_sum <= 0.0001 || alpha_sum <= 0.0001 {
        return shadow_color;
    }
    let out_alpha = (alpha_sum / weight_sum) * uniforms.opacity;
    if out_alpha <= 0.0001 {
        return shadow_color;
    }
    let smear_px = length(scaled_dir * uniforms.target_size);
    return finish_card(
        vec4(
            with_sharp_detail(accum / alpha_sum, base_color, sample_target_uv, crop_bounds_uv, smear_px),
            out_alpha
        ),
        target_uv,
        p,
        crop_bounds_uv,
        shadow_color
    );
}

// The blur taps use the plain sampler (the sharp one per tap costs ~2x at 4K
// for detail the smear erases anyway). The sharp sampler's extra detail at
// this pixel is added back while the smear is still shorter than ~2 px, so a
// zoom that starts or stops moving doesn't flip between the two looks.
fn with_sharp_detail(
    blurred: vec3<f32>,
    sharp: vec4<f32>,
    uv: vec2<f32>,
    crop_bounds_uv: vec4<f32>,
    smear_px: f32
) -> vec3<f32> {
    if uniforms.sampling_params.x <= 1.0 || uniforms.preserve_source_alpha > 0.5 {
        return blurred;
    }
    let keep = 1.0 - smoothstep(0.0, 2.0, smear_px);
    if keep <= 0.0 {
        return blurred;
    }
    let detail = sharp.rgb - sample_texture(uv, crop_bounds_uv).rgb;
    return clamp(blurred + detail * keep, vec3<f32>(0.0), vec3<f32>(1.0));
}

fn finish_card(
    color: vec4<f32>,
    target_uv: vec2<f32>,
    p: vec2<f32>,
    crop_bounds_uv: vec4<f32>,
    shadow_color: vec4<f32>
) -> vec4<f32> {
    let lensed = apply_loupe(color, p, crop_bounds_uv);
    return composite_source_over(apply_color_grade(lensed, target_uv, p), shadow_color);
}

fn card_upscale_ratio(crop_bounds_uv: vec4<f32>) -> f32 {
    let source_size = uniforms.frame_size * (crop_bounds_uv.zw - crop_bounds_uv.xy);
    return max(
        uniforms.target_size.x / max(source_size.x, 1.0),
        uniforms.target_size.y / max(source_size.y, 1.0)
    );
}

// The sharp path cross-fades in over the first 15% of zoom and 10% of
// upsampling so starting or ending a zoom never pops between the two
// samplers.
fn sample_display(uv: vec2<f32>, crop_bounds_uv: vec4<f32>) -> vec4<f32> {
    let display_zoom = uniforms.sampling_params.x;
    if display_zoom > 1.0 && uniforms.preserve_source_alpha < 0.5 {
        let upscale = card_upscale_ratio(crop_bounds_uv);
        let blend = smoothstep(1.0, 1.15, display_zoom) * smoothstep(1.0, 1.1, upscale);
        if blend > 0.0 {
            let sharp = sample_sharp(uv, crop_bounds_uv, upscale, 1.0);
            if blend >= 1.0 {
                return sharp;
            }
            return mix(sample_texture(uv, crop_bounds_uv), sharp, blend);
        }
    }
    return sample_texture(uv, crop_bounds_uv);
}

struct CroppedSample {
    uv: vec2<f32>,
    safe_min: vec2<f32>,
    safe_max: vec2<f32>,
};

fn crop_sample_uv(uv: vec2<f32>, crop_bounds_uv: vec4<f32>) -> CroppedSample {
    var sample_uv = uv;
    if uniforms.mirror_x != 0.0 {
        sample_uv.x = 1.0 - sample_uv.x;
    }
    let crop_size = crop_bounds_uv.zw - crop_bounds_uv.xy;
    let texel_offset = 1.0 / uniforms.frame_size;
    let safe_min = crop_bounds_uv.xy + texel_offset;
    let safe_max = crop_bounds_uv.zw - texel_offset;
    return CroppedSample(
        clamp(sample_uv * crop_size + crop_bounds_uv.xy, safe_min, safe_max),
        safe_min,
        safe_max
    );
}

fn source_alpha(texel: vec4<f32>) -> f32 {
    return select(1.0, texel.a, uniforms.preserve_source_alpha > 0.5);
}

fn sample_bilinear(uv: vec2<f32>, crop_bounds_uv: vec4<f32>) -> vec4<f32> {
    let cropped = crop_sample_uv(clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0)), crop_bounds_uv);
    let texel = textureSampleLevel(frame_texture, frame_sampler, cropped.uv, 0.0);
    return vec4(texel.rgb, source_alpha(texel));
}

fn sample_catmull_rom(uv: vec2<f32>, safe_min: vec2<f32>, safe_max: vec2<f32>) -> vec3<f32> {
    let tex_size = uniforms.frame_size;
    let sample_pos = uv * tex_size;
    let tex_pos1 = floor(sample_pos - 0.5) + 0.5;
    let f = sample_pos - tex_pos1;

    let w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
    let w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
    let w2 = f * (0.5 + f * (2.0 - 1.5 * f));
    let w3 = f * f * (-0.5 + 0.5 * f);
    let w12 = w1 + w2;

    let p0 = clamp((tex_pos1 - 1.0) / tex_size, safe_min, safe_max);
    let p12 = clamp((tex_pos1 + w2 / w12) / tex_size, safe_min, safe_max);
    let p3 = clamp((tex_pos1 + 2.0) / tex_size, safe_min, safe_max);

    var color = vec3<f32>(0.0);
    color += textureSampleLevel(frame_texture, frame_sampler, vec2(p0.x, p0.y), 0.0).rgb * w0.x * w0.y;
    color += textureSampleLevel(frame_texture, frame_sampler, vec2(p12.x, p0.y), 0.0).rgb * w12.x * w0.y;
    color += textureSampleLevel(frame_texture, frame_sampler, vec2(p3.x, p0.y), 0.0).rgb * w3.x * w0.y;
    color += textureSampleLevel(frame_texture, frame_sampler, vec2(p0.x, p12.y), 0.0).rgb * w0.x * w12.y;
    color += textureSampleLevel(frame_texture, frame_sampler, vec2(p12.x, p12.y), 0.0).rgb * w12.x * w12.y;
    color += textureSampleLevel(frame_texture, frame_sampler, vec2(p3.x, p12.y), 0.0).rgb * w3.x * w12.y;
    color += textureSampleLevel(frame_texture, frame_sampler, vec2(p0.x, p3.y), 0.0).rgb * w0.x * w3.y;
    color += textureSampleLevel(frame_texture, frame_sampler, vec2(p12.x, p3.y), 0.0).rgb * w12.x * w3.y;
    color += textureSampleLevel(frame_texture, frame_sampler, vec2(p3.x, p3.y), 0.0).rgb * w3.x * w3.y;
    return color;
}

fn sample_sharp(
    uv: vec2<f32>,
    crop_bounds_uv: vec4<f32>,
    upscale: f32,
    strength: f32
) -> vec4<f32> {
    if uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 {
        return vec4(0.0);
    }
    let cropped = crop_sample_uv(uv, crop_bounds_uv);
    let texel = 1.0 / uniforms.frame_size;
    let center = sample_catmull_rom(cropped.uv, cropped.safe_min, cropped.safe_max);

    let north = textureSampleLevel(
        frame_texture, frame_sampler,
        clamp(cropped.uv - vec2(0.0, texel.y), cropped.safe_min, cropped.safe_max), 0.0
    ).rgb;
    let south = textureSampleLevel(
        frame_texture, frame_sampler,
        clamp(cropped.uv + vec2(0.0, texel.y), cropped.safe_min, cropped.safe_max), 0.0
    ).rgb;
    let west = textureSampleLevel(
        frame_texture, frame_sampler,
        clamp(cropped.uv - vec2(texel.x, 0.0), cropped.safe_min, cropped.safe_max), 0.0
    ).rgb;
    let east = textureSampleLevel(
        frame_texture, frame_sampler,
        clamp(cropped.uv + vec2(texel.x, 0.0), cropped.safe_min, cropped.safe_max), 0.0
    ).rgb;

    let bilinear_sample = textureSampleLevel(frame_texture, frame_sampler, cropped.uv, 0.0);
    let bilinear = bilinear_sample.rgb;
    let lo = min(bilinear, min(min(north, south), min(west, east)));
    let hi = max(bilinear, max(max(north, south), max(west, east)));
    let lobe_strength = SHARPEN_MAX_LOBE * strength *
        smoothstep(1.0, SHARPEN_FULL_UPSCALE, upscale);
    let amp = sqrt(clamp(min(lo, vec3(1.0) - hi) / max(hi, vec3(1e-4)), vec3(0.0), vec3(1.0)));
    let lobe = -amp * lobe_strength;
    let sharpened = (center + (north + south + west + east) * lobe) / (vec3(1.0) + 4.0 * lobe);
    return vec4(clamp(sharpened, lo, hi), source_alpha(bilinear_sample));
}

fn card_shape_coverage(p: vec2<f32>) -> f32 {
    let center = (uniforms.target_bounds.xy + uniforms.target_bounds.zw) * 0.5;
    let size = (uniforms.target_bounds.zw - uniforms.target_bounds.xy) * 0.5;
    let distance = sdf_rounded_rect(
        p - center,
        size,
        corner_radius_for(p - center),
        uniforms.rounding_type
    );
    return clamp(0.5 - distance, 0.0, 1.0);
}

fn loupe_sdf(local: vec2<f32>, half_extent: vec2<f32>, corner: f32) -> f32 {
    return sdf_rounded_rect(local, half_extent, corner, uniforms.sampling_params.y);
}

fn loupe_sample_uv(
    p: vec2<f32>,
    lens_center: vec2<f32>,
    anchor: vec2<f32>,
    bend: f32
) -> vec2<f32> {
    let magnification = uniforms.loupe_shape.w;
    let sample_px = anchor + (p - lens_center) * bend / magnification;
    return (sample_px - uniforms.target_bounds.xy) / uniforms.target_size;
}

fn apply_loupe(card: vec4<f32>, p: vec2<f32>, crop_bounds_uv: vec4<f32>) -> vec4<f32> {
    let magnification = uniforms.loupe_shape.w;
    if magnification < 1.0 {
        return card;
    }

    let half_extent = uniforms.loupe_shape.xy;
    let corner = uniforms.loupe_shape.z;
    let radius = min(half_extent.x, half_extent.y);
    let lens_center = uniforms.target_bounds.xy + uniforms.loupe_position.xy * uniforms.target_size;
    let anchor = uniforms.target_bounds.xy + uniforms.loupe_position.zw * uniforms.target_size;
    let local = p - lens_center;

    let shadow_offset = vec2<f32>(0.0, radius * 0.07);
    let shadow_blur = max(radius * 0.2, 2.0);
    let reach = half_extent + vec2<f32>(shadow_blur + shadow_offset.y + 1.0);
    if abs(local.x) > reach.x || abs(local.y) > reach.y {
        return card;
    }

    let card_coverage = card_shape_coverage(p);
    let dist = loupe_sdf(local, half_extent, corner);
    let shadow_dist = loupe_sdf(local - shadow_offset, half_extent, corner);
    let shadow = LOUPE_SHADOW_OPACITY * card_coverage *
        (1.0 - smoothstep(-shadow_blur * 0.4, shadow_blur, shadow_dist));
    let shadowed = vec4<f32>(card.rgb * (1.0 - shadow), card.a);

    let coverage = clamp(0.5 - dist, 0.0, 1.0);
    if coverage <= 0.0 || card_coverage <= 0.0 {
        return shadowed;
    }

    let depth = max(-dist, 0.0);
    let rim_width = max(radius * 0.06, 1.5);
    let interior = clamp(depth / (rim_width * 3.0), 0.0, 1.0);
    let edge = 1.0 - interior;
    let bend = 1.0 + LOUPE_REFRACTION * edge * edge;

    let travel = uniforms.loupe_motion;
    var rgb: vec3<f32>;
    var content_alpha: f32;
    if dot(travel, travel) > 1e-10 {
        let dither = interleaved_noise(p);
        let center_travel = travel.xy * uniforms.target_size;
        let anchor_travel = travel.zw * uniforms.target_size;
        var accum = vec3<f32>(0.0);
        var alpha_sum = 0.0;
        for (var i = 0; i < LOUPE_BLUR_TAPS; i = i + 1) {
            let k = (f32(i) + dither) / f32(LOUPE_BLUR_TAPS);
            let tap = sample_bilinear(
                loupe_sample_uv(p, lens_center - center_travel * k, anchor - anchor_travel * k, bend),
                crop_bounds_uv
            );
            accum += tap.rgb * tap.a;
            alpha_sum += tap.a;
        }
        rgb = accum / max(alpha_sum, 1e-4);
        content_alpha = alpha_sum / f32(LOUPE_BLUR_TAPS);
    } else {
        let upscale = card_upscale_ratio(crop_bounds_uv) * magnification;
        let uv = clamp(
            loupe_sample_uv(p, lens_center, anchor, bend),
            vec2<f32>(0.0),
            vec2<f32>(1.0)
        );
        let content = sample_sharp(uv, crop_bounds_uv, upscale, 1.0);
        rgb = content.rgb;
        content_alpha = content.a;
        if edge > 0.0 {
            let split = LOUPE_CHROMA * edge * edge;
            rgb.r = sample_bilinear(
                loupe_sample_uv(p, lens_center, anchor, bend * (1.0 + split)),
                crop_bounds_uv
            ).r;
            rgb.b = sample_bilinear(
                loupe_sample_uv(p, lens_center, anchor, bend * (1.0 - split)),
                crop_bounds_uv
            ).b;
        }
    }

    let normalized = local / max(half_extent, vec2<f32>(1.0));
    let light_dir = normalize(vec2<f32>(-0.55, -0.83));
    let facing = dot(normalize(local + vec2<f32>(1e-4, 0.0)), light_dir);

    rgb = rgb * (1.0 - 0.14 * edge * edge);
    let sheen = 0.06 * (1.0 - smoothstep(-0.95, 0.15, normalized.y)) * interior;
    rgb = mix(rgb, vec3<f32>(1.0), sheen);

    let ring = 1.0 - smoothstep(0.0, rim_width, depth);
    rgb = mix(rgb, vec3<f32>(1.0), ring * (0.22 + 0.5 * max(facing, 0.0)));
    rgb = mix(rgb, vec3<f32>(0.0), ring * 0.28 * max(-facing, 0.0));
    let outline = 1.0 - smoothstep(0.0, 1.25, abs(dist + 0.6));
    rgb = mix(rgb, vec3<f32>(0.0), outline * 0.35);

    // The glass never widens the card: it takes the card's own coverage
    // (smear included) and only swaps what is seen through it, except where
    // a transparent source lets magnified content show over a hole.
    let card_alpha = card_coverage * uniforms.opacity;
    let base_alpha = select(min(card.a, card_alpha), card_alpha, uniforms.preserve_source_alpha > 0.5);
    let lens_alpha = base_alpha * content_alpha;
    let out_alpha = mix(shadowed.a, lens_alpha, coverage);
    if out_alpha <= 0.0001 {
        return vec4<f32>(0.0);
    }
    let out_rgb = mix(shadowed.rgb * shadowed.a, rgb * lens_alpha, coverage) / out_alpha;
    return vec4<f32>(out_rgb, out_alpha);
}

fn sample_texture(uv: vec2<f32>, crop_bounds_uv: vec4<f32>) -> vec4<f32> {
    if uv.x >= 0.0 && uv.x <= 1.0 && uv.y >= 0.0 && uv.y <= 1.0 {
        var sample_uv = uv;
        if uniforms.mirror_x != 0.0 {
            sample_uv.x = 1.0 - sample_uv.x;
        }

        let crop_size = crop_bounds_uv.zw - crop_bounds_uv.xy;
        var cropped_uv = sample_uv * crop_size + crop_bounds_uv.xy;

        let texel_offset = 1.0 / uniforms.frame_size;
        let safe_min = crop_bounds_uv.xy + texel_offset;
        let safe_max = crop_bounds_uv.zw - texel_offset;
        cropped_uv = clamp(cropped_uv, safe_min, safe_max);

        let source_size = uniforms.frame_size * crop_size;
        let target_size = uniforms.target_size;
        let scale_ratio = source_size / target_size;
        let is_downscaling = max(scale_ratio.x, scale_ratio.y) > 1.1;
        let upscale_ratio = max(target_size.x / source_size.x, target_size.y / source_size.y);
        let is_upscaling = upscale_ratio > 1.05;

        let center_sample = textureSample(frame_texture, frame_sampler, cropped_uv);
        let center_color = center_sample.rgb;
        let out_alpha = select(1.0, center_sample.a, uniforms.preserve_source_alpha > 0.5);

        if uniforms.preserve_source_alpha > 0.5 && center_sample.a < 0.999 {
            if !is_downscaling {
                return center_sample;
            }

            let footprint = scale_ratio / uniforms.frame_size * 0.5;
            let left = textureSampleLevel(
                frame_texture, frame_sampler,
                clamp(cropped_uv - vec2<f32>(footprint.x, 0.0), safe_min, safe_max), 0.0
            );
            let right = textureSampleLevel(
                frame_texture, frame_sampler,
                clamp(cropped_uv + vec2<f32>(footprint.x, 0.0), safe_min, safe_max), 0.0
            );
            let top = textureSampleLevel(
                frame_texture, frame_sampler,
                clamp(cropped_uv - vec2<f32>(0.0, footprint.y), safe_min, safe_max), 0.0
            );
            let bottom = textureSampleLevel(
                frame_texture, frame_sampler,
                clamp(cropped_uv + vec2<f32>(0.0, footprint.y), safe_min, safe_max), 0.0
            );
            let alpha_sum = center_sample.a * 4.0 + left.a + right.a + top.a + bottom.a;
            let color_sum = center_sample.rgb * center_sample.a * 4.0
                + left.rgb * left.a + right.rgb * right.a + top.rgb * top.a + bottom.rgb * bottom.a;
            return vec4<f32>(color_sum / max(alpha_sum, 0.0001), alpha_sum * 0.125);
        }

        if is_downscaling {
            let texel_size = 1.0 / uniforms.frame_size;

            let offset_x = vec2<f32>(texel_size.x, 0.0);
            let offset_y = vec2<f32>(0.0, texel_size.y);

            let left = textureSample(
                frame_texture,
                frame_sampler,
                clamp(cropped_uv - offset_x, safe_min, safe_max)
            ).rgb;
            let right = textureSample(
                frame_texture,
                frame_sampler,
                clamp(cropped_uv + offset_x, safe_min, safe_max)
            ).rgb;
            let top = textureSample(
                frame_texture,
                frame_sampler,
                clamp(cropped_uv - offset_y, safe_min, safe_max)
            ).rgb;
            let bottom = textureSample(
                frame_texture,
                frame_sampler,
                clamp(cropped_uv + offset_y, safe_min, safe_max)
            ).rgb;

            let blurred = (left + right + top + bottom) * 0.25;

            let sharpness = min(scale_ratio.x * 0.3, 0.7);
            let sharpened = center_color + (center_color - blurred) * sharpness;

            return vec4(clamp(sharpened, vec3<f32>(0.0), vec3<f32>(1.0)), out_alpha);
        }

        if is_upscaling {
            let texel_size = 1.0 / uniforms.frame_size;

            let offset_x = vec2<f32>(texel_size.x, 0.0);
            let offset_y = vec2<f32>(0.0, texel_size.y);

            let left = textureSample(
                frame_texture,
                frame_sampler,
                clamp(cropped_uv - offset_x, safe_min, safe_max)
            ).rgb;
            let right = textureSample(
                frame_texture,
                frame_sampler,
                clamp(cropped_uv + offset_x, safe_min, safe_max)
            ).rgb;
            let top = textureSample(
                frame_texture,
                frame_sampler,
                clamp(cropped_uv - offset_y, safe_min, safe_max)
            ).rgb;
            let bottom = textureSample(
                frame_texture,
                frame_sampler,
                clamp(cropped_uv + offset_y, safe_min, safe_max)
            ).rgb;

            let blurred = (left + right + top + bottom) * 0.25;
            let sharpness = min((upscale_ratio - 1.0) * 0.25, 0.45);
            let sharpened = center_color + (center_color - blurred) * sharpness;

            return vec4(clamp(sharpened, vec3<f32>(0.0), vec3<f32>(1.0)), out_alpha);
        }

        return vec4(center_color, out_alpha);
    }

    return vec4(0.0);
}

fn apply_rounded_corners(current_color: vec4<f32>, target_uv: vec2<f32>) -> vec4<f32> {
    let centered_uv = (target_uv - vec2<f32>(0.5)) * uniforms.target_size;
    let half_size = uniforms.target_size * 0.5;
    let coverage = rounded_rect_coverage(
        centered_uv,
        half_size,
        corner_radius_for(centered_uv),
        uniforms.rounding_type
    );

    return vec4(current_color.rgb, current_color.a * coverage);
}

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}
