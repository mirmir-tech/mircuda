/// Real (SASS) architecture NVRTC compiles for. Kernels are compiled on the
/// device that runs them, so PTX portability buys nothing, and the driver
/// cannot JIT PTX from a toolkit newer than itself.
pub(super) fn target_architecture(capability: (i32, i32)) -> String {
    if capability.0 == 12 {
        return "sm_120f".into();
    }
    format!("sm_{}{}", capability.0, capability.1)
}

#[cfg(test)]
mod tests {
    use super::target_architecture;

    #[test]
    fn targets_sm12_devices_as_a_feature_family() {
        assert_eq!(target_architecture((12, 0)), "sm_120f");
        assert_eq!(target_architecture((12, 1)), "sm_120f");
    }

    #[test]
    fn targets_the_exact_architecture_of_earlier_devices() {
        assert_eq!(target_architecture((8, 9)), "sm_89");
        assert_eq!(target_architecture((10, 0)), "sm_100");
    }
}
