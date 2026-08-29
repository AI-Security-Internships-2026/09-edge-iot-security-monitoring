from opacus.accountants.utils import get_noise_multiplier

dataset_size = 53000  # <-- REPLACE with your real per-client partition size

alphas = [1 + x / 10.0 for x in range(1, 100)] + list(range(11, 200))

for epochs in [5, 3, 2, 1]:
    for batch_size in [512, 256, 128, 64, 32]:
        try:
            sigma = get_noise_multiplier(
                target_epsilon=0.05,
                target_delta=1e-5,
                sample_rate=batch_size / 400000,
                epochs=epochs,
                accountant="rdp",
                alphas=alphas,
            )
            print(f"epochs={epochs} batch_size={batch_size}: sigma={sigma:.3f}  -> FEASIBLE")
        except Exception as e:
            print(f"epochs={epochs} batch_size={batch_size}: FAILED ({e})")