pub(super) fn target_architecture(capability: (i32, i32)) -> String {
    if capability.0 == 12 {
        return "compute_120f".into();
    }
    format!("compute_{}{}", capability.0, capability.1)
}

#[cfg(test)]
mod tests {
    use super::target_architecture;

    #[test]
    fn targets_sm12_devices_as_a_feature_family() {
        assert_eq!(target_architecture((12, 0)), "compute_120f");
        assert_eq!(target_architecture((12, 1)), "compute_120f");
    }

    #[test]
    fn preserves_virtual_architecture_for_earlier_devices() {
        assert_eq!(target_architecture((8, 9)), "compute_89");
        assert_eq!(target_architecture((10, 0)), "compute_100");
    }
}
