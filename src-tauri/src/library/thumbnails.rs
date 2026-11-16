use crate::app::settings::{AppSettings, load_settings};
use crate::app::state::{AppState, ThumbnailSource};
use crate::decode::as_shot::as_shot;
use crate::decode::formats::is_raw_file;
use crate::decode::image_loader;
use crate::library::file_ops::{is_cloud_placeholder, read_file_mapped};
use crate::library::listing::enqueue_metadata;
use crate::library::sidecars::{parse_virtual_path, run_batch_workers};
use crate::masks::generation::MaskDefinition;
use crate::metadata::sidecar::{ImageMetadata, is_image_edited};
use crate::render::adjustment_utils::effective_baseline_exposure;
use crate::render::geometry::{
    Crop, apply_coarse_rotation, get_geometry_params_from_json, is_geometry_identity,
};
use crate::render::gpu::context::{GpuContext, get_or_init_gpu_context};
use crate::render::gpu::output::process_and_get_dynamic_image;
use crate::render::gpu::processor::RenderRequest;
use crate::render::image_processing::{apply_cpu_default_raw_processing, apply_exposure};
use crate::render::options::{resolve_render_options, resolve_tonemapper_override};
use crate::render::source::{Frame, RenderSource, SourceTransform};
use crate::render::uniforms::get_all_adjustments_from_json;
use anyhow::Result;
use image::codecs::jpeg::JpegEncoder;
use image::{DynamicImage, GenericImageView};
use std::borrow::Cow;
use std::fs;
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::Ordering;
use tauri::{AppHandle, Emitter, Manager};

#[tauri::command]
pub fn cancel_thumbnail_generation(
    state: tauri::State<AppState>,
    app_handle: tauri::AppHandle,
) -> Result<(), String> {
    state
        .thumbnail_cancellation_token
        .store(true, Ordering::SeqCst);

    let mut tracker = state.thumbnail_progress.lock().unwrap();
    tracker.total = 0;
    tracker.completed = 0;
    drop(tracker);

    let _ = app_handle.emit(
        "thumbnail-progress",
        serde_json::json!({ "current": 0, "total": 0 }),
    );
    Ok(())
}

pub fn resolve_thumbnail_cache_dir(app_handle: &AppHandle) -> std::result::Result<PathBuf, String> {
    let cache_dir = app_handle
        .path()
        .app_cache_dir()
        .map_err(|e| e.to_string())?;
    let thumb_cache_dir = cache_dir.join("thumbnails");
    if !thumb_cache_dir.exists() {
        fs::create_dir_all(&thumb_cache_dir).map_err(|e| e.to_string())?;
    }
    Ok(thumb_cache_dir)
}

pub fn emit_thumbnail_cache_setup_error(app_handle: &AppHandle, path: &str, reason: &str) {
    let _ = app_handle.emit(
        "thumbnail-generation-error",
        serde_json::json!({ "path": path, "reason": reason }),
    );
}

// The key leaves out render settings, so changing them doesn't regenerate the
// whole library's thumbnails.
fn compute_thumbnail_cache_hash(path_str: &str, adjustments_bytes: &[u8]) -> Option<String> {
    let (source_path, _) = parse_virtual_path(path_str);

    let img_mod_time = fs::metadata(&source_path)
        .ok()?
        .modified()
        .ok()?
        .duration_since(std::time::UNIX_EPOCH)
        .ok()?
        .as_secs();

    let mut hasher = blake3::Hasher::new();
    hasher.update(path_str.as_bytes());
    hasher.update(&img_mod_time.to_le_bytes());
    hasher.update(adjustments_bytes);
    Some(hasher.finalize().to_hex().to_string())
}

const THUMBNAIL_SOURCE_CACHE_ENTRIES: usize = 8;

const THUMBNAIL_SOURCE_CACHE_BYTES: usize = 256 << 20;

/// Unedited raws (or only default adjustments, no real crop, no lens blur) can
/// show the camera's embedded JPEG instead of a decode.
fn can_use_embedded_preview(
    adjustments: &serde_json::Value,
    source_path: &Path,
    settings: &AppSettings,
    file_bytes: &[u8],
) -> bool {
    // Custom develop defaults (per-camera presets) only show in a rendered thumbnail.
    if crate::adjustments::develop_defaults::is_customized(source_path) {
        return false;
    }
    if adjustments.is_null() {
        return true;
    }
    let tm = resolve_tonemapper_override(settings, true);
    if is_image_edited(adjustments, source_path, true, tm)
        || adjustments["lensBlurEnabled"].as_bool().unwrap_or(false)
    {
        return false;
    }
    let Ok(crop) = serde_json::from_value::<Crop>(adjustments["crop"].clone()) else {
        return true;
    };
    let Some((w, h, _)) = crate::decode::raw_processing::get_raw_dimensions(file_bytes) else {
        return false;
    };
    let near = |a: f64, b: u32| (a - b as f64).abs() <= b as f64 * 0.01 + 1.0;
    (near(crop.width, w) && near(crop.height, h)) || (near(crop.width, h) && near(crop.height, w))
}

fn has_ai_patches(adjustments: &serde_json::Value) -> bool {
    adjustments["aiPatches"]
        .as_array()
        .is_some_and(|patches| !patches.is_empty())
}

/// For a linear DNG, the smallest proxy size that still covers the thumbnail
/// (or its crop) at `target_res`.
fn thumbnail_proxy_min_dim(
    file_bytes: &[u8],
    target_res: u32,
    crop: Option<&Crop>,
) -> Option<usize> {
    let (full_w, full_h, is_linear) =
        crate::decode::raw_processing::get_raw_dimensions(file_bytes)?;
    if !is_linear {
        return None;
    }
    let full_max_dim = full_w.max(full_h) as f64;
    let needed = match crop {
        Some(c) if c.width > 0.0 && c.height > 0.0 => {
            target_res as f64 * full_max_dim / c.width.max(c.height)
        }
        _ => target_res as f64,
    };
    Some(needed.min(full_max_dim).ceil() as usize)
}

pub fn generate_thumbnail_data(
    path_str: &str,
    gpu_context: Option<&GpuContext>,
    preloaded_image: Option<&DynamicImage>,
    app_handle: &AppHandle,
) -> anyhow::Result<DynamicImage> {
    let metadata = read_thumbnail_metadata(path_str, app_handle);
    let settings = load_settings(app_handle.clone()).unwrap_or_default();
    generate_thumbnail_data_with_metadata(
        path_str,
        gpu_context,
        preloaded_image,
        app_handle,
        metadata.as_ref(),
        &settings,
    )
}

fn read_thumbnail_metadata(path_str: &str, app_handle: &AppHandle) -> Option<ImageMetadata> {
    let (source_path, sidecar_path) = parse_virtual_path(path_str);
    if is_cloud_placeholder(&sidecar_path) {
        enqueue_metadata(
            app_handle,
            path_str.to_string(),
            source_path.clone(),
            sidecar_path.clone(),
        );
        None
    } else {
        let mut metadata = fs::read_to_string(&sidecar_path)
            .ok()
            .and_then(|content| crate::metadata::sidecar::parse_sidecar(&content));
        resolve_unedited_metadata(&mut metadata, path_str);
        metadata
    }
}

fn metadata_adjustments_bytes(metadata: &Option<ImageMetadata>) -> Vec<u8> {
    metadata
        .as_ref()
        .map(|meta| serde_json::to_vec(&meta.adjustments).unwrap_or_default())
        .unwrap_or_default()
}

fn resolve_unedited_metadata(metadata: &mut Option<ImageMetadata>, path_str: &str) -> bool {
    let (source_path, _) = parse_virtual_path(path_str);
    if !crate::adjustments::develop_defaults::is_customized(&source_path) {
        return false;
    }
    let meta = metadata.get_or_insert_with(ImageMetadata::default);
    if !crate::render::adjustment_utils::is_unedited(&meta.adjustments) {
        return false;
    }
    crate::adjustments::develop_defaults::resolve_in_place(&mut meta.adjustments, &source_path);
    true
}

fn generate_thumbnail_data_with_metadata(
    path_str: &str,
    gpu_context: Option<&GpuContext>,
    preloaded_image: Option<&DynamicImage>,
    app_handle: &AppHandle,
    metadata: Option<&ImageMetadata>,
    settings: &AppSettings,
) -> anyhow::Result<DynamicImage> {
    let (source_path, _) = parse_virtual_path(path_str);
    let source_path_str = source_path.to_string_lossy().to_string();
    let is_raw = is_raw_file(&source_path_str);

    let null_adjustments = serde_json::Value::Null;
    let adjustments = metadata.map_or(&null_adjustments, |m| &m.adjustments);
    let always_decode_raw = settings.always_decode_raw_thumbnails.unwrap_or(false);

    // The smallest embedded JPEG that covers the medium thumbnail, if it is at
    // least about the small thumbnail size.
    if is_raw
        && preloaded_image.is_none()
        && !always_decode_raw
        && let Ok(mmap) = read_file_mapped(&source_path)
        && can_use_embedded_preview(adjustments, &source_path, settings, &mmap)
        && let Some(preview) = image_loader::safe_embedded_preview(
            &mmap,
            &source_path_str,
            Some(settings.medium_thumbnail_resolution.unwrap_or(1280)),
        )
        && preview.width().max(preview.height())
            >= (settings.small_thumbnail_resolution.unwrap_or(480) as f32 * 0.95) as u32
    {
        return Ok(preview);
    }

    if let (Some(context), Some(meta)) = (gpu_context, metadata)
        && !meta.adjustments.is_null()
    {
        let state = app_handle.state::<AppState>();
        let target_res = settings.medium_thumbnail_resolution.unwrap_or(1280);

        let base_cache_hash = crate::render::cache_utils::calculate_thumbnail_source_hash(
            path_str,
            &meta.adjustments,
        );

        let crop_data: Option<Crop> = serde_json::from_value(meta.adjustments["crop"].clone()).ok();

        let thumbnail_transform = |base: &ThumbnailSource| {
            let geometry = get_geometry_params_from_json(path_str, &meta.adjustments);
            let warp = (!is_geometry_identity(&geometry)).then_some(geometry);
            let mut transform = SourceTransform {
                warp,
                working_fit: Some((base.processing_dim, base.processing_dim)),
                effects: true,
                ..Default::default()
            };
            let (w, h) = transform.working_size(base.image.dimensions());
            let scale = if base.full_max_dim > 0 {
                w.max(h) as f32 / base.full_max_dim as f32
            } else {
                1.0
            } * base.raw_scale_factor;
            transform.frame = Frame::from_adjustments(&meta.adjustments).scale_crop(scale as f64);
            (transform, scale)
        };

        let cached_base: Option<Arc<ThumbnailSource>> = {
            let cache = state.thumbnail_source_cache.lock().unwrap();
            cache
                .get(path_str)
                .filter(|base| {
                    let (_, scale) = thumbnail_transform(base);
                    let sufficient_resolution = crop_data
                        .as_ref()
                        .filter(|c| c.width > 0.0 && c.height > 0.0)
                        .is_none_or(|c| {
                            (c.width as f32 * scale).max(c.height as f32 * scale)
                                >= target_res as f32 * 0.95
                        });
                    base.hash == base_cache_hash && sufficient_resolution
                })
                .cloned()
        };

        let base = if let Some(base) = cached_base {
            base
        } else {
            let mut raw_scale_factor: f32;

            let composite_image: Cow<DynamicImage> = if let Some(img) = preloaded_image {
                let prepared =
                    image_loader::prepare_base_image_cow(img, &source_path_str, adjustments)?;
                raw_scale_factor = crate::ai::upscale::scale_between(img, &prepared);
                prepared
            } else {
                let mmap_guard;
                let vec_guard;

                let file_slice: &[u8] = match read_file_mapped(&source_path) {
                    Ok(mmap) => {
                        mmap_guard = Some(mmap);
                        mmap_guard.as_ref().unwrap()
                    }
                    Err(e) => {
                        if preloaded_image.is_none() {
                            log::warn!("Fallback read for {}: {}", source_path_str, e);
                        }
                        let bytes = fs::read(&source_path).map_err(|io_err| {
                            anyhow::anyhow!(
                                "Fallback read failed for {}: {}",
                                source_path_str,
                                io_err
                            )
                        })?;
                        vec_guard = Some(bytes);
                        vec_guard.as_ref().unwrap()
                    }
                };

                crate::decode::as_shot::as_shot_from_bytes(&source_path_str, file_slice);
                let proxy_min_dim = if is_raw && !has_ai_patches(adjustments) {
                    thumbnail_proxy_min_dim(file_slice, target_res, crop_data.as_ref())
                } else {
                    None
                };
                let (img, upscale) = image_loader::load_and_composite_with_proxy(
                    file_slice,
                    &source_path_str,
                    adjustments,
                    true,
                    settings,
                    None,
                    proxy_min_dim,
                )?;

                raw_scale_factor = upscale;
                if is_raw {
                    raw_scale_factor *=
                        crate::decode::raw_processing::get_fast_demosaic_scale_factor(
                            file_slice,
                            (img.width() as f32 / upscale).round() as u32,
                            (img.height() as f32 / upscale).round() as u32,
                        );
                }
                Cow::Owned(img)
            };

            let full_max_dim = composite_image.width().max(composite_image.height());
            let texture_limit = context.limits.max_texture_dimension_2d;

            let mut processing_dim = target_res;
            if let Some(c) = &crop_data
                && c.width > 0.0
                && c.height > 0.0
            {
                let crop_max_dim_loaded = c.width.max(c.height) * raw_scale_factor as f64;
                if crop_max_dim_loaded > 0.0 {
                    processing_dim = ((target_res as f64 * full_max_dim as f64
                        / crop_max_dim_loaded)
                        .round() as u32)
                        .min(full_max_dim);
                }
            }

            let processing_dim = processing_dim.min(texture_limit);

            // Geometry and effects run on the full-size image and the result is
            // downscaled to `processing_dim` at the end; only an image larger
            // than the GPU allows is reduced first.
            let working_image = if full_max_dim > texture_limit {
                crate::render::image_processing::downscale_f32_image(
                    &composite_image,
                    texture_limit,
                    texture_limit,
                )
            } else {
                composite_image.into_owned()
            };

            let base = Arc::new(ThumbnailSource {
                hash: base_cache_hash,
                image: Arc::new(working_image),
                full_max_dim,
                processing_dim,
                raw_scale_factor,
            });
            let mut cache = state.thumbnail_source_cache.lock().unwrap();
            cache.remove(path_str);
            let bytes = |entry: &ThumbnailSource| entry.image.as_bytes().len();
            let mut total: usize = cache.values().map(|entry| bytes(entry)).sum();
            while !cache.is_empty()
                && (cache.len() >= THUMBNAIL_SOURCE_CACHE_ENTRIES
                    || total + bytes(&base) > THUMBNAIL_SOURCE_CACHE_BYTES)
            {
                let key = cache.keys().next().cloned().unwrap();
                total -= cache.remove(&key).map_or(0, |entry| bytes(&entry));
            }
            cache.insert(path_str.to_string(), Arc::clone(&base));
            base
        };

        let (transform, total_scale) = thumbnail_transform(&base);
        let source_id = {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            std::hash::Hash::hash(&(path_str, base.hash), &mut hasher);
            std::hash::Hasher::finish(&hasher)
        };
        let source = RenderSource::new(&base.image, source_id, &meta.adjustments, transform);
        let (preview_w, preview_h) = source.size();
        let unscaled_crop_offset = crop_data.map_or((0.0, 0.0), |c| (c.x as f32, c.y as f32));

        let mask_definitions: Vec<MaskDefinition> = meta
            .adjustments
            .get("masks")
            .and_then(|m| serde_json::from_value(m.clone()).ok())
            .unwrap_or_else(Vec::new);

        let mask_bitmaps = crate::masks::generation::get_cached_or_generate_masks(
            &state,
            path_str,
            &mask_definitions,
            preview_w,
            preview_h,
            total_scale,
            (
                unscaled_crop_offset.0 * total_scale,
                unscaled_crop_offset.1 * total_scale,
            ),
            &meta.adjustments,
        )
        .bitmaps;

        let gpu_adjustments = get_all_adjustments_from_json(
            &meta.adjustments,
            is_raw,
            as_shot(&source_path_str),
            resolve_render_options(settings, is_raw),
        );
        let lut_path = meta.adjustments["lutPath"].as_str();
        let lut = lut_path.and_then(|p| {
            let mut cache = state.lut_cache.lock().unwrap();
            if let Some(cached_lut) = cache.get(p) {
                return Some(cached_lut.clone());
            }
            if let Ok(loaded_lut) = crate::adjustments::lut_processing::parse_lut_file(p) {
                let arc_lut = Arc::new(loaded_lut);
                cache.insert(p.to_string(), arc_lut.clone());
                return Some(arc_lut);
            }
            None
        });

        let processed_image = process_and_get_dynamic_image(
            context,
            &state,
            &source,
            RenderRequest {
                adjustments: gpu_adjustments,
                mask_bitmaps: &mask_bitmaps,
                mask_cache_key: None,
                lut,
                roi: None,
                hsl_probe: false,
                motion_blur: crate::effects::motion_blur::MotionBlurs::new(
                    path_str,
                    &meta.adjustments,
                    (preview_w, preview_h),
                    (
                        unscaled_crop_offset.0 * total_scale,
                        unscaled_crop_offset.1 * total_scale,
                    ),
                    {
                        let (w, h) = source.transform.working_size(base.image.dimensions());
                        w.max(h) as f32
                    },
                ),
            },
            "generate_thumbnail_data",
        );
        return processed_image.map_err(|e| anyhow::anyhow!(e));
    }

    let mut final_image: Cow<DynamicImage> = if let Some(img) = preloaded_image {
        image_loader::prepare_base_image_cow(img, &source_path_str, adjustments)?
    } else {
        let mmap_guard;
        let vec_guard;
        let file_slice: &[u8] = match read_file_mapped(&source_path) {
            Ok(mmap) => {
                mmap_guard = mmap;
                &mmap_guard
            }
            Err(e) => {
                log::warn!("Fallback read for {}: {}", source_path_str, e);
                vec_guard = fs::read(&source_path)?;
                &vec_guard
            }
        };
        if adjustments.is_null() {
            crate::decode::as_shot::as_shot_from_bytes(&source_path_str, file_slice);
        }
        Cow::Owned(
            image_loader::load_and_composite(
                file_slice,
                &source_path_str,
                adjustments,
                true,
                settings,
                None,
            )?
            .0,
        )
    };

    if adjustments.is_null() {
        let render = resolve_render_options(settings, is_raw);
        let mut owned_image = apply_exposure(
            final_image.into_owned(),
            effective_baseline_exposure(adjustments, as_shot(&source_path_str), render.baseline),
        );
        let use_agx = render.tonemapper_override == Some(1);

        if use_agx {
            if !is_raw {
                owned_image = crate::render::image_processing::apply_srgb_to_linear(owned_image);
            }
            crate::render::image_processing::apply_cpu_agx_tonemap(&mut owned_image);
        } else if is_raw {
            apply_cpu_default_raw_processing(&mut owned_image);
        }
        final_image = Cow::Owned(owned_image);
    }

    let fallback_orientation_steps = adjustments["orientationSteps"].as_u64().unwrap_or(0) as u8;
    Ok(apply_coarse_rotation(final_image, fallback_orientation_steps).into_owned())
}

fn encode_thumbnail(image: &DynamicImage, target_width: u32) -> Result<Vec<u8>> {
    let thumbnail =
        crate::render::image_processing::downscale_f32_image(image, target_width, target_width);
    let mut buf = Cursor::new(Vec::new());
    let mut encoder = JpegEncoder::new_with_quality(&mut buf, 75);
    encoder.encode_image(&thumbnail.to_rgb8())?;
    Ok(buf.into_inner())
}

#[allow(clippy::too_many_arguments)]
pub fn generate_single_thumbnail_and_cache(
    path_str: &str,
    thumb_cache_dir: &Path,
    gpu_context: Option<&GpuContext>,
    preloaded_image: Option<&DynamicImage>,
    force_regenerate: bool,
    app_handle: &AppHandle,
    settings: &AppSettings,
    before_generate: Option<&dyn Fn(bool)>,
) -> Option<(String, String, u8, bool)> {
    let (source_path, sidecar_path) = parse_virtual_path(path_str);
    let sidecar_is_placeholder = is_cloud_placeholder(&sidecar_path);

    let mut metadata: Option<ImageMetadata> = None;
    let (rating, is_edited, mut adjustments_bytes) = if sidecar_is_placeholder {
        enqueue_metadata(
            app_handle,
            path_str.to_string(),
            source_path.clone(),
            sidecar_path.clone(),
        );
        (0, false, Vec::new())
    } else if let Ok(content) = fs::read_to_string(&sidecar_path) {
        if let Some(meta) = crate::metadata::sidecar::parse_sidecar(&content) {
            let is_raw = crate::decode::formats::is_raw_file(path_str);
            let tm = resolve_tonemapper_override(settings, is_raw);
            let info = (
                meta.rating,
                is_image_edited(&meta.adjustments, &source_path, is_raw, tm),
                serde_json::to_vec(&meta.adjustments).unwrap_or_default(),
            );
            metadata = Some(meta);
            info
        } else {
            (0, false, Vec::new())
        }
    } else {
        (0, false, Vec::new())
    };

    if !sidecar_is_placeholder && resolve_unedited_metadata(&mut metadata, path_str) {
        adjustments_bytes = metadata_adjustments_bytes(&metadata);
    }

    if let Some(remote) =
        crate::integrations::immich::placeholder_thumbnail(app_handle, path_str, thumb_cache_dir)
    {
        return remote.map(|(small, medium)| (small, medium, rating, is_edited));
    }

    let cache_hash = compute_thumbnail_cache_hash(path_str, &adjustments_bytes)?;

    let small_path = thumb_cache_dir.join(format!("{}_small.jpg", cache_hash));
    let medium_path = thumb_cache_dir.join(format!("{}_medium.jpg", cache_hash));

    if !force_regenerate && small_path.exists() && medium_path.exists() {
        crate::app::storage::mark_used(&small_path);
        return Some((
            small_path.to_string_lossy().into_owned(),
            medium_path.to_string_lossy().into_owned(),
            rating,
            is_edited,
        ));
    }

    if let Some(before_generate) = before_generate {
        before_generate(is_edited);
    }

    if is_cloud_placeholder(&source_path) {
        return None;
    }

    let target_width_small = settings.small_thumbnail_resolution.unwrap_or(480);
    let target_width_medium = settings.medium_thumbnail_resolution.unwrap_or(1280);

    let thumb_result = if sidecar_is_placeholder {
        generate_thumbnail_data(path_str, gpu_context, preloaded_image, app_handle)
    } else {
        let current_settings = load_settings(app_handle.clone()).unwrap_or_default();
        generate_thumbnail_data_with_metadata(
            path_str,
            gpu_context,
            preloaded_image,
            app_handle,
            metadata.as_ref(),
            &current_settings,
        )
    };

    if let Ok(thumb_image) = thumb_result
        && let (Ok(small_data), Ok(medium_data)) = (
            encode_thumbnail(&thumb_image, target_width_small),
            encode_thumbnail(&thumb_image, target_width_medium),
        )
    {
        let _ = fs::write(&small_path, &small_data);
        let _ = fs::write(&medium_path, &medium_data);
        crate::app::storage::record_thumbnail_write((small_data.len() + medium_data.len()) as u64);
        return Some((
            small_path.to_string_lossy().into_owned(),
            medium_path.to_string_lossy().into_owned(),
            rating,
            is_edited,
        ));
    }
    None
}

const RAW_HEADER_PREFETCH_BYTES: u64 = 8 * 1024 * 1024;

/// Reads the source ahead into the OS cache on rotational disks. For a raw
/// whose thumbnail comes from data near the start (embedded preview, or a
/// linear DNG proxy), only the first 8 MB are read.
fn prefetch_source_file(path_str: &str, is_edited: bool, always_decode_raw: bool) {
    use std::io::Read;
    let (source_path, _) = parse_virtual_path(path_str);
    let Ok(mut file) = std::fs::File::open(&source_path) else {
        return;
    };
    let source_path_str = source_path.to_string_lossy().into_owned();
    if is_raw_file(&source_path_str) {
        let mut head = Vec::new();
        if (&mut file)
            .take(RAW_HEADER_PREFETCH_BYTES)
            .read_to_end(&mut head)
            .is_err()
        {
            return;
        }
        if !is_edited && !always_decode_raw {
            return;
        }
        let is_dng = source_path_str.to_lowercase().ends_with(".dng");
        let is_linear_dng = is_dng
            && std::panic::catch_unwind(|| {
                crate::decode::raw_processing::get_raw_dimensions(&head)
            })
            .ok()
            .flatten()
            .is_some_and(|(_, _, is_linear)| is_linear);
        if is_linear_dng && !always_decode_raw {
            return;
        }
    }
    let _ = std::io::copy(&mut file, &mut std::io::sink());
}

pub fn start_thumbnail_workers(app_handle: tauri::AppHandle) {
    let state = app_handle.state::<AppState>();
    let manager = state.thumbnail_manager.clone();
    let settings = load_settings(app_handle.clone()).unwrap_or_default();
    let thread_count = settings.thumbnail_worker_threads.unwrap_or(4).clamp(1, 16);

    for _ in 0..thread_count {
        let app_clone = app_handle.clone();
        let manager_clone = manager.clone();

        std::thread::spawn(move || {
            loop {
                let path_to_process: String = {
                    let mut queue = manager_clone.queue.lock().unwrap();
                    while queue.is_empty() {
                        queue = manager_clone.cvar.wait(queue).unwrap();
                    }
                    let path = queue.pop_back().unwrap();

                    let mut processing = manager_clone.processing_now.lock().unwrap();
                    if processing.contains(&path) {
                        let state = app_clone.state::<AppState>();
                        increment_thumbnail_progress(&state, &app_clone);
                        continue;
                    }
                    processing.insert(path.clone());
                    path
                };

                let state = app_clone.state::<AppState>();
                let gpu_context = get_or_init_gpu_context(&state, &app_clone).ok();

                let current_settings = load_settings(app_clone.clone()).unwrap_or_default();

                if let Ok(cache_dir) = resolve_thumbnail_cache_dir(&app_clone) {
                    let always_decode_raw = current_settings
                        .always_decode_raw_thumbnails
                        .unwrap_or(false);
                    let prefetch = |is_edited: bool| {
                        if manager_clone.rotational_disk.load(Ordering::Relaxed) {
                            let _io_permit = manager_clone.io_gate.lock().unwrap();
                            prefetch_source_file(&path_to_process, is_edited, always_decode_raw);
                        }
                    };

                    let result = generate_single_thumbnail_and_cache(
                        &path_to_process,
                        &cache_dir,
                        gpu_context.as_ref(),
                        None,
                        false,
                        &app_clone,
                        &current_settings,
                        Some(&prefetch),
                    );

                    if let Some((small_path, medium_path, rating, is_edited)) = result {
                        emit_thumbnail_generated(
                            &app_clone,
                            &path_to_process,
                            &small_path,
                            &medium_path,
                            rating,
                            is_edited,
                        );
                    }
                    increment_thumbnail_progress(&state, &app_clone);
                }
                manager_clone
                    .processing_now
                    .lock()
                    .unwrap()
                    .remove(&path_to_process);
            }
        });
    }
}

const MAX_QUEUED_THUMBNAILS: usize = 500;

#[tauri::command]
pub fn update_thumbnail_queue(
    paths: Vec<String>,
    app_handle: tauri::AppHandle,
) -> Result<(), String> {
    let state = app_handle.state::<AppState>();

    let mut queue = state.thumbnail_manager.queue.lock().unwrap();

    if paths.is_empty() {
        queue.clear();
        let mut tracker = state.thumbnail_progress.lock().unwrap();
        tracker.total = 0;
        tracker.completed = 0;
        drop(tracker);

        let _ = app_handle.emit(
            "thumbnail-progress",
            serde_json::json!({ "current": 0, "total": 0 }),
        );
        state.thumbnail_manager.cvar.notify_all();
        return Ok(());
    }

    let mut unique_paths = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for path in paths {
        if seen.insert(path.clone()) {
            unique_paths.push(path);
        }
    }

    queue.retain(|p| !seen.contains(p));

    unique_paths.truncate(MAX_QUEUED_THUMBNAILS);
    while queue.len() + unique_paths.len() > MAX_QUEUED_THUMBNAILS && queue.pop_front().is_some() {}

    if state
        .thumbnail_manager
        .rotational_disk
        .load(Ordering::Relaxed)
    {
        unique_paths.sort();
        for path in unique_paths.into_iter().rev() {
            queue.push_back(path);
        }
    } else {
        for path in unique_paths {
            queue.push_back(path);
        }
    }

    let queue_len = queue.len();
    drop(queue);

    let mut tracker = state.thumbnail_progress.lock().unwrap();
    tracker.total = tracker.completed + queue_len;

    let current = tracker.completed;
    let total = tracker.total;
    drop(tracker);

    let _ = app_handle.emit(
        "thumbnail-progress",
        serde_json::json!({ "current": current, "total": total }),
    );

    state.thumbnail_manager.cvar.notify_all();
    Ok(())
}

pub fn add_to_thumbnail_queue(state: &AppState, count: usize, app_handle: &AppHandle) {
    let mut tracker = state.thumbnail_progress.lock().unwrap();
    tracker.total += count;
    let current = tracker.completed;
    let total = tracker.total;
    drop(tracker);

    let _ = app_handle.emit(
        "thumbnail-progress",
        serde_json::json!({ "current": current, "total": total }),
    );
}

pub fn increment_thumbnail_progress(state: &AppState, app_handle: &AppHandle) {
    let mut tracker = state.thumbnail_progress.lock().unwrap();
    tracker.completed += 1;
    let current = tracker.completed;
    let total = tracker.total;

    if current >= total {
        tracker.total = 0;
        tracker.completed = 0;
        drop(tracker);

        let _ = app_handle.emit(
            "thumbnail-progress",
            serde_json::json!({ "current": 0, "total": 0 }),
        );
        let _ = app_handle.emit("thumbnail-generation-complete", true);
    } else {
        drop(tracker);
        let _ = app_handle.emit(
            "thumbnail-progress",
            serde_json::json!({ "current": current, "total": total }),
        );
    }
}

pub fn emit_thumbnail_generated(
    app_handle: &AppHandle,
    path: &str,
    small_thumbnail_path: &str,
    medium_thumbnail_path: &str,
    rating: u8,
    is_edited: bool,
) {
    let _ = app_handle.emit(
        "thumbnail-generated",
        serde_json::json!({
            "path": path,
            "thumbnailPath": small_thumbnail_path,
            "previewPath": medium_thumbnail_path,
            "rating": rating,
            "is_edited": is_edited
        }),
    );
}

#[tauri::command]
pub fn clear_thumbnail_cache(app_handle: AppHandle) -> Result<(), String> {
    let cache_dir = app_handle
        .path()
        .app_cache_dir()
        .map_err(|e| e.to_string())?;
    let thumb_cache_dir = cache_dir.join("thumbnails");

    if thumb_cache_dir.exists() {
        fs::remove_dir_all(&thumb_cache_dir)
            .map_err(|e| format!("Failed to remove thumbnail cache: {}", e))?;
    }

    fs::create_dir_all(&thumb_cache_dir)
        .map_err(|e| format!("Failed to recreate thumbnail cache directory: {}", e))?;
    crate::app::storage::reset_thumbnail_estimate();

    Ok(())
}

pub fn get_cache_key_hash(path_str: &str) -> Option<String> {
    let (_, sidecar_path) = parse_virtual_path(path_str);

    let mut metadata = fs::read_to_string(&sidecar_path)
        .ok()
        .and_then(|content| crate::metadata::sidecar::parse_sidecar(&content));
    let mut adjustments_bytes = metadata_adjustments_bytes(&metadata);
    if resolve_unedited_metadata(&mut metadata, path_str) {
        adjustments_bytes = metadata_adjustments_bytes(&metadata);
    }

    compute_thumbnail_cache_hash(path_str, &adjustments_bytes)
}

pub fn get_cached_or_generate_thumbnail_image(
    path_str: &str,
    app_handle: &AppHandle,
    gpu_context: Option<&GpuContext>,
) -> Result<DynamicImage> {
    let thumb_cache_dir =
        resolve_thumbnail_cache_dir(app_handle).map_err(|e| anyhow::anyhow!(e))?;
    let settings = load_settings(app_handle.clone()).unwrap_or_default();
    let target_width_small = settings.small_thumbnail_resolution.unwrap_or(480);
    let target_width_medium = settings.medium_thumbnail_resolution.unwrap_or(1280);

    if let Some(cache_hash) = get_cache_key_hash(path_str) {
        let cache_path = thumb_cache_dir.join(format!("{}_medium.jpg", cache_hash));

        if cache_path.exists() {
            if let Ok(image) = image::open(&cache_path) {
                crate::app::storage::mark_used(&cache_path);
                return Ok(image);
            }
            eprintln!(
                "Could not open cached thumbnail, regenerating: {:?}",
                cache_path
            );
        }

        let thumb_image = generate_thumbnail_data(path_str, gpu_context, None, app_handle)?;
        if let (Ok(small_data), Ok(medium_data)) = (
            encode_thumbnail(&thumb_image, target_width_small),
            encode_thumbnail(&thumb_image, target_width_medium),
        ) {
            let _ = fs::write(
                thumb_cache_dir.join(format!("{}_small.jpg", cache_hash)),
                &small_data,
            );
            let _ = fs::write(
                thumb_cache_dir.join(format!("{}_medium.jpg", cache_hash)),
                &medium_data,
            );
            crate::app::storage::record_thumbnail_write(
                (small_data.len() + medium_data.len()) as u64,
            );
        }

        Ok(thumb_image)
    } else {
        generate_thumbnail_data(path_str, gpu_context, None, app_handle)
    }
}

pub(crate) fn regenerate_thumbnails(
    paths: impl Iterator<Item = String> + Send,
    app_handle: &AppHandle,
    settings: &AppSettings,
    force_regenerate: bool,
) {
    let state = app_handle.state::<AppState>();
    let thumb_cache_dir = match resolve_thumbnail_cache_dir(app_handle) {
        Ok(dir) => dir,
        Err(e) => {
            log::warn!("Unable to initialize thumbnail cache directory: {}", e);
            for path in paths {
                emit_thumbnail_cache_setup_error(app_handle, &path, &e);
                increment_thumbnail_progress(&state, app_handle);
            }
            return;
        }
    };

    let gpu_context = get_or_init_gpu_context(&state, app_handle).ok();

    run_batch_workers(paths, |path| {
        let path_str = path.as_str();
        let result = generate_single_thumbnail_and_cache(
            path_str,
            &thumb_cache_dir,
            gpu_context.as_ref(),
            None,
            force_regenerate,
            app_handle,
            settings,
            None,
        );

        if let Some((small_path, medium_path, rating, is_edited)) = result {
            emit_thumbnail_generated(
                app_handle,
                path_str,
                &small_path,
                &medium_path,
                rating,
                is_edited,
            );
        }

        increment_thumbnail_progress(&state, app_handle);
    });
}
