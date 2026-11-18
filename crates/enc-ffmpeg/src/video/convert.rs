use ffmpeg::{format::Pixel, frame, software::scaling};

pub(crate) enum VideoConverter {
    Swscale(scaling::Context),
    Rgb(RgbToYuv420),
}

impl VideoConverter {
    pub fn run(
        &mut self,
        input: &frame::Video,
        output: &mut frame::Video,
    ) -> Result<(), ffmpeg::Error> {
        match self {
            Self::Swscale(context) => context.run(input, output),
            Self::Rgb(converter) => converter.run(input, output),
        }
    }
}

const LUMA_SHIFT: i32 = 14;
const CHROMA_SHIFT: i32 = 16;
const LUMA_OFFSET: i32 = (16 << LUMA_SHIFT) + (1 << (LUMA_SHIFT - 1));
const CHROMA_OFFSET: i32 = (128 << CHROMA_SHIFT) + (1 << (CHROMA_SHIFT - 1));
const LUMA: [i32; 3] = [2991, 10064, 1016];
const CB: [i32; 3] = [-1649, -5547, 7196];
const CR: [i32; 3] = [7196, -6536, -660];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Layout {
    Bgrx,
    Rgbx,
}

#[derive(Clone, Copy, Debug)]
pub struct RgbToYuv420 {
    input: Pixel,
    output: Pixel,
    width: u32,
    height: u32,
    layout: Layout,
    simd: bool,
}

enum ChromaRow<'a> {
    Planar(&'a mut [u8], &'a mut [u8]),
    Interleaved(&'a mut [u8]),
}

impl RgbToYuv420 {
    pub fn new(input: Pixel, output: Pixel, width: u32, height: u32) -> Option<Self> {
        let layout = match input {
            Pixel::BGRA | Pixel::BGRZ => Layout::Bgrx,
            Pixel::RGBA | Pixel::RGBZ => Layout::Rgbx,
            _ => return None,
        };
        if !matches!(output, Pixel::YUV420P | Pixel::NV12)
            || width == 0
            || height == 0
            || !width.is_multiple_of(2)
            || !height.is_multiple_of(2)
        {
            return None;
        }
        Some(Self {
            input,
            output,
            width,
            height,
            layout,
            simd: simd_available(),
        })
    }

    pub fn is_accelerated(&self) -> bool {
        self.simd
    }

    pub fn without_simd(mut self) -> Self {
        self.simd = false;
        self
    }

    pub fn run(
        &self,
        input: &frame::Video,
        output: &mut frame::Video,
    ) -> Result<(), ffmpeg::Error> {
        if input.format() != self.input
            || input.width() != self.width
            || input.height() != self.height
        {
            return Err(ffmpeg::Error::InputChanged);
        }
        if output.format() != self.output
            || output.width() != self.width
            || output.height() != self.height
        {
            *output = frame::Video::new(self.output, self.width, self.height);
        }

        let width = self.width as usize;
        let height = self.height as usize;
        let source_stride = input.stride(0);
        let source = input.data(0);
        let planar = self.output == Pixel::YUV420P;
        let mut planes = output_planes(output, if planar { 3 } else { 2 });
        let (luma, chroma) = planes.split_at_mut(1);
        let (luma, luma_stride) = (&mut *luma[0].0, luma[0].1);

        for row in 0..height / 2 {
            let top = &source[2 * row * source_stride..][..width * 4];
            let bottom = &source[(2 * row + 1) * source_stride..][..width * 4];
            let (luma_top, luma_bottom) = luma[2 * row * luma_stride..].split_at_mut(luma_stride);
            let luma_top = &mut luma_top[..width];
            let luma_bottom = &mut luma_bottom[..width];
            let chroma_row = if planar {
                let (cb, cr) = chroma.split_at_mut(1);
                ChromaRow::Planar(
                    &mut cb[0].0[row * cb[0].1..][..width / 2],
                    &mut cr[0].0[row * cr[0].1..][..width / 2],
                )
            } else {
                ChromaRow::Interleaved(&mut chroma[0].0[row * chroma[0].1..][..width])
            };
            self.convert_rows(top, bottom, luma_top, luma_bottom, chroma_row);
        }

        Ok(())
    }

    fn convert_rows(
        &self,
        top: &[u8],
        bottom: &[u8],
        luma_top: &mut [u8],
        luma_bottom: &mut [u8],
        chroma: ChromaRow<'_>,
    ) {
        #[cfg(target_arch = "x86_64")]
        if self.simd {
            unsafe {
                avx2::convert_rows(self.layout, top, bottom, luma_top, luma_bottom, chroma);
            }
            return;
        }
        match self.layout {
            Layout::Bgrx => {
                convert_rows_scalar::<2, 1, 0>(top, bottom, luma_top, luma_bottom, chroma, 0)
            }
            Layout::Rgbx => {
                convert_rows_scalar::<0, 1, 2>(top, bottom, luma_top, luma_bottom, chroma, 0)
            }
        }
    }
}

fn output_planes(output: &mut frame::Video, count: usize) -> Vec<(&mut [u8], usize)> {
    let planes: Vec<(*mut u8, usize, usize)> = (0..count)
        .map(|plane| {
            let stride = output.stride(plane);
            let data = output.data_mut(plane);
            (data.as_mut_ptr(), data.len(), stride)
        })
        .collect();
    planes
        .into_iter()
        .map(|(pointer, len, stride)| {
            (
                unsafe { std::slice::from_raw_parts_mut(pointer, len) },
                stride,
            )
        })
        .collect()
}

#[cfg(target_arch = "x86_64")]
fn simd_available() -> bool {
    std::arch::is_x86_feature_detected!("avx2")
}

#[cfg(not(target_arch = "x86_64"))]
fn simd_available() -> bool {
    false
}

#[inline(always)]
fn luma(r: i32, g: i32, b: i32) -> u8 {
    ((LUMA[0] * r + LUMA[1] * g + LUMA[2] * b + LUMA_OFFSET) >> LUMA_SHIFT).clamp(0, 255) as u8
}

#[inline(always)]
fn chroma(coefficients: [i32; 3], r: i32, g: i32, b: i32) -> u8 {
    ((coefficients[0] * r + coefficients[1] * g + coefficients[2] * b + CHROMA_OFFSET)
        >> CHROMA_SHIFT)
        .clamp(0, 255) as u8
}

#[inline(always)]
fn convert_rows_scalar<const R: usize, const G: usize, const B: usize>(
    top: &[u8],
    bottom: &[u8],
    luma_top: &mut [u8],
    luma_bottom: &mut [u8],
    chroma_row: ChromaRow<'_>,
    start: usize,
) {
    for (source, target) in [(top, &mut *luma_top), (bottom, &mut *luma_bottom)] {
        for (pixel, value) in source[start * 4..]
            .chunks_exact(4)
            .zip(target[start..].iter_mut())
        {
            *value = luma(pixel[R] as i32, pixel[G] as i32, pixel[B] as i32);
        }
    }

    let pairs = top[start * 4..]
        .chunks_exact(8)
        .zip(bottom[start * 4..].chunks_exact(8))
        .map(|(a, b)| {
            let r = a[R] as i32 + a[4 + R] as i32 + b[R] as i32 + b[4 + R] as i32;
            let g = a[G] as i32 + a[4 + G] as i32 + b[G] as i32 + b[4 + G] as i32;
            let b = a[B] as i32 + a[4 + B] as i32 + b[B] as i32 + b[4 + B] as i32;
            (chroma(CB, r, g, b), chroma(CR, r, g, b))
        });
    match chroma_row {
        ChromaRow::Planar(cb, cr) => {
            for ((u, v), (cb, cr)) in
                pairs.zip(cb[start / 2..].iter_mut().zip(cr[start / 2..].iter_mut()))
            {
                *cb = u;
                *cr = v;
            }
        }
        ChromaRow::Interleaved(uv) => {
            for ((u, v), target) in pairs.zip(uv[start..].chunks_exact_mut(2)) {
                target[0] = u;
                target[1] = v;
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use super::{
        CB, CHROMA_OFFSET, CHROMA_SHIFT, CR, ChromaRow, LUMA, LUMA_OFFSET, LUMA_SHIFT, Layout,
        convert_rows_scalar,
    };
    use std::arch::x86_64::*;

    const BLOCK: usize = 16;

    fn weights(layout: Layout, coefficients: [i32; 3]) -> [i16; 16] {
        let [r, g, b] = coefficients.map(|value| value as i16);
        let pixel = match layout {
            Layout::Bgrx => [b, g, r, 0],
            Layout::Rgbx => [r, g, b, 0],
        };
        let mut weights = [0i16; 16];
        for (index, weight) in weights.iter_mut().enumerate() {
            *weight = pixel[index % 4];
        }
        weights
    }

    #[target_feature(enable = "avx2")]
    unsafe fn load(bytes: &[u8]) -> __m256i {
        unsafe { _mm256_loadu_si256(bytes.as_ptr().cast()) }
    }

    #[target_feature(enable = "avx2")]
    fn luma8(pixels: __m256i, weights: __m256i) -> __m256i {
        let zero = _mm256_setzero_si256();
        let low = _mm256_madd_epi16(_mm256_unpacklo_epi8(pixels, zero), weights);
        let high = _mm256_madd_epi16(_mm256_unpackhi_epi8(pixels, zero), weights);
        _mm256_srai_epi32::<LUMA_SHIFT>(_mm256_add_epi32(
            _mm256_hadd_epi32(low, high),
            _mm256_set1_epi32(LUMA_OFFSET),
        ))
    }

    #[target_feature(enable = "avx2")]
    fn luma16(first: __m256i, second: __m256i, weights: __m256i) -> __m128i {
        let packed = _mm256_permute4x64_epi64::<0xD8>(_mm256_packs_epi32(
            luma8(first, weights),
            luma8(second, weights),
        ));
        _mm_packus_epi16(
            _mm256_castsi256_si128(packed),
            _mm256_extracti128_si256::<1>(packed),
        )
    }

    #[target_feature(enable = "avx2")]
    fn quad_sums(top: __m256i, bottom: __m256i) -> __m256i {
        let zero = _mm256_setzero_si256();
        let low = _mm256_add_epi16(
            _mm256_unpacklo_epi8(top, zero),
            _mm256_unpacklo_epi8(bottom, zero),
        );
        let high = _mm256_add_epi16(
            _mm256_unpackhi_epi8(top, zero),
            _mm256_unpackhi_epi8(bottom, zero),
        );
        let low = _mm256_add_epi16(low, _mm256_srli_si256::<8>(low));
        let high = _mm256_add_epi16(high, _mm256_srli_si256::<8>(high));
        _mm256_unpacklo_epi64(low, high)
    }

    #[target_feature(enable = "avx2")]
    fn chroma8(first: __m256i, second: __m256i, weights: __m256i) -> __m256i {
        let sums = _mm256_hadd_epi32(
            _mm256_madd_epi16(first, weights),
            _mm256_madd_epi16(second, weights),
        );
        let ordered = _mm256_permutevar8x32_epi32(sums, _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7));
        _mm256_srai_epi32::<CHROMA_SHIFT>(_mm256_add_epi32(
            ordered,
            _mm256_set1_epi32(CHROMA_OFFSET),
        ))
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn convert_rows(
        layout: Layout,
        top: &[u8],
        bottom: &[u8],
        luma_top: &mut [u8],
        luma_bottom: &mut [u8],
        chroma_row: ChromaRow<'_>,
    ) {
        let luma_weights = weights(layout, LUMA);
        let cb_weights = weights(layout, CB);
        let cr_weights = weights(layout, CR);
        let (luma_weights, cb_weights, cr_weights) = unsafe {
            (
                _mm256_loadu_si256(luma_weights.as_ptr().cast()),
                _mm256_loadu_si256(cb_weights.as_ptr().cast()),
                _mm256_loadu_si256(cr_weights.as_ptr().cast()),
            )
        };
        let width = luma_top.len();
        let blocks = width / BLOCK;
        let mut chroma_row = chroma_row;

        for block in 0..blocks {
            let x = block * BLOCK;
            let (top0, top1, bottom0, bottom1) = unsafe {
                (
                    load(&top[x * 4..x * 4 + 32]),
                    load(&top[x * 4 + 32..x * 4 + 64]),
                    load(&bottom[x * 4..x * 4 + 32]),
                    load(&bottom[x * 4 + 32..x * 4 + 64]),
                )
            };
            let luma_top_block = luma16(top0, top1, luma_weights);
            let luma_bottom_block = luma16(bottom0, bottom1, luma_weights);

            let first = quad_sums(top0, bottom0);
            let second = quad_sums(top1, bottom1);
            let cb = chroma8(first, second, cb_weights);
            let cr = chroma8(first, second, cr_weights);
            let packed = _mm256_permute4x64_epi64::<0xD8>(_mm256_packs_epi32(cb, cr));
            let cbcr = _mm_packus_epi16(
                _mm256_castsi256_si128(packed),
                _mm256_extracti128_si256::<1>(packed),
            );

            let luma_top_target = &mut luma_top[x..x + BLOCK];
            let luma_bottom_target = &mut luma_bottom[x..x + BLOCK];
            unsafe {
                _mm_storeu_si128(luma_top_target.as_mut_ptr().cast(), luma_top_block);
                _mm_storeu_si128(luma_bottom_target.as_mut_ptr().cast(), luma_bottom_block);
            }
            match &mut chroma_row {
                ChromaRow::Planar(cb_row, cr_row) => {
                    let cb_target = &mut cb_row[x / 2..x / 2 + 8];
                    let cr_target = &mut cr_row[x / 2..x / 2 + 8];
                    unsafe {
                        _mm_storel_epi64(cb_target.as_mut_ptr().cast(), cbcr);
                        _mm_storel_epi64(cr_target.as_mut_ptr().cast(), _mm_srli_si128::<8>(cbcr));
                    }
                }
                ChromaRow::Interleaved(uv_row) => {
                    let target = &mut uv_row[x..x + BLOCK];
                    let interleaved = _mm_unpacklo_epi8(cbcr, _mm_srli_si128::<8>(cbcr));
                    unsafe {
                        _mm_storeu_si128(target.as_mut_ptr().cast(), interleaved);
                    }
                }
            }
        }

        let start = blocks * BLOCK;
        if start == width {
            return;
        }
        match layout {
            Layout::Bgrx => convert_rows_scalar::<2, 1, 0>(
                top,
                bottom,
                luma_top,
                luma_bottom,
                chroma_row,
                start,
            ),
            Layout::Rgbx => convert_rows_scalar::<0, 1, 2>(
                top,
                bottom,
                luma_top,
                luma_bottom,
                chroma_row,
                start,
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn source(format: Pixel, width: u32, height: u32, seed: usize) -> frame::Video {
        let mut frame = frame::Video::new(format, width, height);
        let stride = frame.stride(0);
        let data = frame.data_mut(0);
        for y in 0..height as usize {
            for x in 0..width as usize * 4 {
                data[y * stride + x] = ((x * 31 + y * 17 + seed * 7 + (x * y) % 13) % 256) as u8;
            }
        }
        frame
    }

    fn channels(format: Pixel) -> [usize; 3] {
        match format {
            Pixel::BGRA | Pixel::BGRZ => [2, 1, 0],
            _ => [0, 1, 2],
        }
    }

    fn sample(frame: &frame::Video, plane: usize, x: usize, y: usize) -> u8 {
        frame.data(plane)[y * frame.stride(plane) + x]
    }

    fn chroma_samples(frame: &frame::Video, x: usize, y: usize) -> (u8, u8) {
        if frame.format() == Pixel::NV12 {
            (sample(frame, 1, x * 2, y), sample(frame, 1, x * 2 + 1, y))
        } else {
            (sample(frame, 1, x, y), sample(frame, 2, x, y))
        }
    }

    fn reference(input: &frame::Video, x: usize, y: usize) -> [f64; 3] {
        let [r, g, b] = channels(input.format());
        let offset = y * input.stride(0) + x * 4;
        let pixel = &input.data(0)[offset..offset + 4];
        [pixel[r] as f64, pixel[g] as f64, pixel[b] as f64]
    }

    #[test]
    fn matches_bt709_limited_range_reference() {
        let (kr, kg, kb) = (0.2126, 0.7152, 0.0722);
        for input_format in [Pixel::BGRA, Pixel::BGRZ, Pixel::RGBA, Pixel::RGBZ] {
            for output_format in [Pixel::YUV420P, Pixel::NV12] {
                let input = source(input_format, 62, 22, 3);
                let converter = RgbToYuv420::new(input_format, output_format, 62, 22).unwrap();
                let mut output = frame::Video::empty();
                converter.run(&input, &mut output).unwrap();
                assert_eq!(output.format(), output_format);

                for y in 0..22 {
                    for x in 0..62 {
                        let [r, g, b] = reference(&input, x, y);
                        let expected = 16.0 + (kr * r + kg * g + kb * b) * 219.0 / 255.0;
                        let actual = sample(&output, 0, x, y) as f64;
                        assert!(
                            (actual - expected).abs() <= 1.0,
                            "luma {x},{y}: {actual} vs {expected}"
                        );
                    }
                }
                for y in 0..11 {
                    for x in 0..31 {
                        let mut sum = [0.0; 3];
                        for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                            let rgb = reference(&input, x * 2 + dx, y * 2 + dy);
                            for channel in 0..3 {
                                sum[channel] += rgb[channel] / 4.0;
                            }
                        }
                        let [r, g, b] = sum;
                        let luma = kr * r + kg * g + kb * b;
                        let cb = 128.0 + (b - luma) / (2.0 * (1.0 - kb)) * 224.0 / 255.0;
                        let cr = 128.0 + (r - luma) / (2.0 * (1.0 - kr)) * 224.0 / 255.0;
                        let (actual_cb, actual_cr) = chroma_samples(&output, x, y);
                        assert!(
                            (actual_cb as f64 - cb).abs() <= 1.0,
                            "cb {x},{y}: {actual_cb} vs {cb}"
                        );
                        assert!(
                            (actual_cr as f64 - cr).abs() <= 1.0,
                            "cr {x},{y}: {actual_cr} vs {cr}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn simd_and_portable_paths_are_bit_identical() {
        for (width, height) in [(16, 2), (48, 8), (1920, 16), (98, 6), (2, 2), (34, 4)] {
            for input_format in [Pixel::BGRZ, Pixel::RGBA] {
                for output_format in [Pixel::YUV420P, Pixel::NV12] {
                    let input = source(input_format, width, height, width as usize);
                    let simd =
                        RgbToYuv420::new(input_format, output_format, width, height).unwrap();
                    let portable = simd.without_simd();
                    let mut simd_output = frame::Video::empty();
                    let mut portable_output = frame::Video::empty();
                    simd.run(&input, &mut simd_output).unwrap();
                    portable.run(&input, &mut portable_output).unwrap();
                    for plane in 0..simd_output.planes() {
                        let row_bytes = if plane == 0 || output_format == Pixel::NV12 {
                            width as usize
                        } else {
                            width as usize / 2
                        };
                        for row in 0..simd_output.plane_height(plane) as usize {
                            let a = &simd_output.data(plane)[row * simd_output.stride(plane)..]
                                [..row_bytes];
                            let b = &portable_output.data(plane)
                                [row * portable_output.stride(plane)..][..row_bytes];
                            assert_eq!(
                                a, b,
                                "{width}x{height} {input_format:?} {output_format:?} plane {plane} row {row}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn rejects_unsupported_configurations_and_mismatched_frames() {
        assert!(RgbToYuv420::new(Pixel::RGB24, Pixel::YUV420P, 64, 64).is_none());
        assert!(RgbToYuv420::new(Pixel::BGRA, Pixel::YUV444P, 64, 64).is_none());
        assert!(RgbToYuv420::new(Pixel::BGRA, Pixel::YUV420P, 63, 64).is_none());
        let converter = RgbToYuv420::new(Pixel::BGRA, Pixel::YUV420P, 64, 64).unwrap();
        let mut output = frame::Video::empty();
        assert!(
            converter
                .run(&source(Pixel::RGBA, 64, 64, 0), &mut output)
                .is_err()
        );
        assert!(
            converter
                .run(&source(Pixel::BGRA, 32, 64, 0), &mut output)
                .is_err()
        );
    }
}
