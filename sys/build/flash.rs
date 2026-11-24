//! Pinned `FlashAttention` sources, staged and patched before compilation.

use std::{
    error::Error,
    fs,
    path::{Path, PathBuf},
};

use super::sources::flash_attn_dir;

const SPECIALIZATIONS: [&str; 6] = [
    "flash_fwd_split_hdim64_bf16_causal_sm80.cu",
    "flash_fwd_split_hdim128_bf16_causal_sm80.cu",
    "flash_fwd_split_hdim256_bf16_causal_sm80.cu",
    "flash_fwd_split_hdim64_bf16_sm80.cu",
    "flash_fwd_split_hdim128_bf16_sm80.cu",
    "flash_fwd_split_hdim256_bf16_sm80.cu",
];

/// One exact edit of a pinned source: `file`, the `count` occurrences of
/// `from` it must contain, and their replacement.
struct Patch {
    file: &'static str,
    from: &'static str,
    to: &'static str,
    count: usize,
}

/// Packing a GQA group's query heads into the row axis (`seqlenq_ngroups_swapped`)
/// shares each K/V load across the group. In decode every packed row is the same
/// query token, so `rows_share_position` makes the window mask and the first K/V
/// block use the last query position for all rows instead of each row's index.
const PATCHES: [Patch; 7] = [
    Patch {
        file: "flash.h",
        from: "(b, ngroups, nheads_kv, d).\n};",
        to: "(b, ngroups, nheads_kv, d).\n    bool rows_share_position;  // Swapped rows are one \
             query token.\n};",
        count: 1,
    },
    Patch {
        file: "mask.h",
        from: "    const float alibi_slope;\n",
        to: "    const float alibi_slope;\n    const bool rows_share_position;\n",
        count: 1,
    },
    Patch {
        file: "mask.h",
        from: "const float alibi_slope=0.f)",
        to: "const float alibi_slope=0.f, const bool rows_share_position=false)",
        count: 1,
    },
    Patch {
        file: "mask.h",
        from: ", alibi_slope(!Has_alibi ? 0.0 : alibi_slope) {",
        to: ", alibi_slope(!Has_alibi ? 0.0 : alibi_slope)\n        , \
             rows_share_position(rows_share_position) {",
        count: 1,
    },
    Patch {
        file: "mask.h",
        from: "\n                        const int row_idx = row_idx_base + i * 8;",
        to: "\n                        const int row_idx = rows_share_position ? max_seqlen_q - 1 : \
             row_idx_base + i * 8;",
        count: 1,
    },
    Patch {
        file: "flash_fwd_kernel.h",
        from: "params.window_size_left, params.window_size_right, alibi_slope);",
        to: "params.window_size_left, params.window_size_right, alibi_slope, \
             params.rows_share_position);",
        count: 2,
    },
    // The first K/V block of a split: shared rows start at the last query row.
    Patch {
        file: "flash_fwd_kernel.h",
        from: ": std::max(n_split_idx * n_blocks_per_split, (m_block * kBlockM + \
               binfo.actual_seqlen_k",
        to: ": std::max(n_split_idx * n_blocks_per_split, ((params.rows_share_position ? \
             binfo.actual_seqlen_q - 1 : m_block * kBlockM) + binfo.actual_seqlen_k",
        count: 1,
    },
];

pub fn compile(configure: &impl Fn(&mut cc::Build), out: &Path) -> Result<(), Box<dyn Error>> {
    let flash = flash_attn_dir()?;
    let cutlass = flash.join("csrc/cutlass/include");
    if !cutlass.join("cutlass/cutlass.h").is_file() {
        return Err(
            format!("pinned FlashAttention sources not found in {}", flash.display()).into()
        );
    }
    let source = stage(&flash.join("csrc/flash_attn/src"), &out.join("flash_attn"))?;
    let mut build = cc::Build::new();
    configure(&mut build);
    build
        .include("native/compat")
        .include(&source)
        .include(&cutlass)
        .flag("--expt-extended-lambda")
        .flag("--use_fast_math")
        .flag("-include")
        .flag("native/compat/flash_attn_prelude.h")
        .define("CUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL", "1")
        .define("FLASHATTENTION_DISABLE_BACKWARD", None)
        .define("FLASHATTENTION_DISABLE_DROPOUT", None)
        .define("FLASHATTENTION_DISABLE_PYBIND", None)
        .files(SPECIALIZATIONS.map(|file| source.join(file)))
        .file("native/flash_attn2_bf16.cu")
        .compile("mircuda_flash_attn2");
    Ok(())
}

/// Copies the kernel headers and used specializations next to each other, since
/// quoted includes resolve beside the including file, then applies [`PATCHES`].
fn stage(source: &Path, staged: &Path) -> Result<PathBuf, Box<dyn Error>> {
    fs::create_dir_all(staged)?;
    for entry in fs::read_dir(source)? {
        let path = entry?.path();
        let name = path.file_name().and_then(|name| name.to_str()).unwrap_or_default();
        let header =
            path.extension().is_some_and(|extension| extension == "h" || extension == "cuh");
        if header || SPECIALIZATIONS.contains(&name) {
            fs::copy(&path, staged.join(name))?;
        }
    }
    for name in SPECIALIZATIONS {
        if !staged.join(name).is_file() {
            return Err(format!("pinned FlashAttention source {name} not found").into());
        }
    }
    for patch in &PATCHES {
        let path = staged.join(patch.file);
        let text = fs::read_to_string(&path)?;
        let found = text.matches(patch.from).count();
        if found != patch.count {
            return Err(format!(
                "FlashAttention patch for {} expected {} match(es) of {:?}, found {found}",
                patch.file, patch.count, patch.from
            )
            .into());
        }
        fs::write(&path, text.replace(patch.from, patch.to))?;
    }
    Ok(staged.to_path_buf())
}
