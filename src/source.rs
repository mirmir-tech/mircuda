use std::borrow::Cow;

/// CUDA source embedded in the Rust binary for NVRTC compilation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KernelSource {
    name: &'static str,
    body: KernelSourceBody,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum KernelSourceBody {
    Single(&'static str),
    Parts(&'static [&'static str]),
}

impl KernelSource {
    /// Creates an inline CUDA translation unit validated by [`crate::cuda_kernel`].
    #[must_use]
    pub const fn inline(source: &'static str) -> Self {
        Self {
            name: "inline.cu",
            body: KernelSourceBody::Single(source),
        }
    }

    /// Creates a translation unit embedded with [`crate::cuda_kernel_file`].
    #[must_use]
    pub const fn embedded(name: &'static str, source: &'static str) -> Self {
        Self {
            name,
            body: KernelSourceBody::Single(source),
        }
    }

    /// Creates one translation unit from ordered, embedded source fragments.
    #[must_use]
    pub const fn composed(name: &'static str, parts: &'static [&'static str]) -> Self {
        Self {
            name,
            body: KernelSourceBody::Parts(parts),
        }
    }

    /// Returns the diagnostic name passed to the compiler.
    #[must_use]
    pub const fn name(self) -> &'static str {
        self.name
    }

    /// Returns the complete CUDA source.
    #[must_use]
    pub fn source(self) -> Cow<'static, str> {
        match self.body {
            KernelSourceBody::Single(source) => Cow::Borrowed(source),
            KernelSourceBody::Parts(parts) => Cow::Owned(parts.concat()),
        }
    }
}
