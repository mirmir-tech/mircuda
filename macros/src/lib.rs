#![deny(missing_docs)]

//! Checked CUDA source and kernel-signature macros re-exported by `mircuda`.

use proc_macro::TokenStream;
use quote::quote;
use syn::{
    Ident, LitStr, Pat, PatType, Token, Type, Visibility, parenthesized,
    parse::{Parse, ParseStream},
    parse_macro_input,
    punctuated::Punctuated,
};

/// Embeds a non-empty inline CUDA translation unit containing a global kernel.
///
/// The macro performs an early structural check for a `__global__` declaration.
/// NVRTC performs complete CUDA compilation later through `mircuda::Compiler`.
/// Use `mircuda::cuda_kernel_file!` or `mircuda::cuda_kernel_files!` for source
/// stored in files.
///
/// # Example
///
/// ```rust,ignore
/// let source = mircuda::cuda_kernel!(r#"
///     extern "C" __global__ void fill(float* output, float value) {
///         output[threadIdx.x] = value;
///     }
/// "#);
///
/// assert_eq!(source.name(), "inline.cu");
/// ```
#[proc_macro]
pub fn cuda_kernel(input: TokenStream) -> TokenStream {
    let source = parse_macro_input!(input as LitStr);
    let value = source.value();
    if value.trim().is_empty() {
        return syn::Error::new(source.span(), "CUDA source cannot be empty")
            .to_compile_error()
            .into();
    }
    if !value.contains("__global__") {
        return syn::Error::new(source.span(), "CUDA source must declare a __global__ kernel")
            .to_compile_error()
            .into();
    }
    quote!(::mircuda::KernelSource::inline(#source)).into()
}

/// Declares a typed Rust contract for one exported CUDA kernel symbol.
///
/// The declaration has the form `Signature = "symbol"(arguments...)`. Immutable
/// device buffers use shared references, writable buffers use mutable references,
/// and CUDA scalar arguments are passed by value. The macro generates a signature
/// type implementing `mircuda::KernelSignature`; resolve it with
/// `mircuda::Module::kernel` and launch it with a tuple in declaration order.
///
/// The declared Rust argument list must exactly match the compiled CUDA symbol's
/// ABI. Keeping the declaration next to its embedded source makes that contract
/// reviewable without exposing raw CUDA arguments.
///
/// # Example
///
/// ```rust,ignore
/// use mircuda::{
///     CompileOptions, Compiler, DeviceBuffer, Driver, LaunchConfig, Result,
///     cuda_export, cuda_kernel,
/// };
///
/// cuda_export!(Fill = "fill"(
///     output: &mut DeviceBuffer<f32>,
///     value: f32,
/// ));
///
/// fn main() -> Result<()> {
///     let source = cuda_kernel!(r#"
///         extern "C" __global__ void fill(float* output, float value) {
///             output[threadIdx.x] = value;
///         }
///     "#);
///     let driver = Driver::initialize()?;
///     let device = driver.devices()?.into_iter().next().expect("CUDA device");
///     let context = driver.create_context(device)?;
///     let stream = context.create_stream()?;
///     let pool = context.default_memory_pool()?;
///     let compiler = Compiler::new(context)?;
///     let module = compiler.compile(source, &CompileOptions::default())?;
///     let kernel = module.kernel::<Fill>()?;
///     let mut output = pool.allocate::<f32>(&stream, 256)?;
///
///     kernel.launch(
///         &stream,
///         LaunchConfig::for_elements(256, 256)?,
///         (&mut output, 2.0),
///     )?;
///     stream.synchronize()?;
///     Ok(())
/// }
/// ```
#[proc_macro]
pub fn cuda_export(input: TokenStream) -> TokenStream {
    let declaration = parse_macro_input!(input as ExportDeclaration);
    match declaration.expand() {
        Ok(tokens) => tokens.into(),
        Err(error) => error.to_compile_error().into(),
    }
}

struct ExportDeclaration {
    visibility: Visibility,
    signature: Ident,
    name: LitStr,
    arguments: Punctuated<PatType, Token![,]>,
}

impl Parse for ExportDeclaration {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        let visibility = input.parse()?;
        let signature = input.parse()?;
        input.parse::<Token![=]>()?;
        let name = input.parse()?;
        let content;
        parenthesized!(content in input);
        let arguments = content.parse_terminated(PatType::parse, Token![,])?;
        Ok(Self { visibility, signature, name, arguments })
    }
}

impl ExportDeclaration {
    fn expand(self) -> syn::Result<proc_macro2::TokenStream> {
        let Self { visibility, signature, name, arguments } = self;
        let mut names = Vec::with_capacity(arguments.len());
        let mut types = Vec::with_capacity(arguments.len());
        let mut encoders = Vec::with_capacity(arguments.len());
        for argument in arguments {
            let Pat::Ident(pattern) = *argument.pat else {
                return Err(syn::Error::new_spanned(argument.pat, "expected an argument name"));
            };
            let identifier = pattern.ident;
            let (argument_type, encoder) = expand_type(&identifier, &argument.ty);
            names.push(identifier);
            types.push(argument_type);
            encoders.push(encoder);
        }
        Ok(quote! {
            #[doc = concat!("Typed CUDA signature for the `", #name, "` symbol.")]
            #visibility struct #signature;

            unsafe impl ::mircuda::KernelSignature for #signature {
                const NAME: &'static str = #name;
                type Arguments<'a> = (#(#types,)*);

                fn encode(arguments: Self::Arguments<'_>) -> ::mircuda::KernelArguments<'_> {
                    let (#(#names,)*) = arguments;
                    let encoded = ::mircuda::KernelArguments::new();
                    #(#encoders)*
                    encoded
                }
            }
        })
    }
}

fn expand_type(name: &Ident, ty: &Type) -> (proc_macro2::TokenStream, proc_macro2::TokenStream) {
    if let Type::Reference(reference) = ty {
        let element = &reference.elem;
        if reference.mutability.is_some() {
            return (quote!(&'a mut #element), quote!(let encoded = encoded.write(#name);));
        }
        return (quote!(&'a #element), quote!(let encoded = encoded.read(#name);));
    }
    (quote!(#ty), quote!(let encoded = encoded.scalar(#name);))
}
