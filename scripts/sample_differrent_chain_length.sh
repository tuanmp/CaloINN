

uv run python scripts/run_mcmc.py  \
    --cinn-ckpt /pscratch/sd/p/pmtuan/caloinn/caloinn_lightning/53921130/checkpoints/epoch=491-val_loss=895.95.ckpt   \
    --clf-ckpt /pscratch/sd/p/pmtuan/caloxtreme_clf/artifacts/best-54873288-val_loss=0.609829-epoch-epoch=999.ckpt     \
    --calibrator temperature_calibration.json     \
    --cinn-config params/pions_odd_sharded.yaml     \
    --n-steps 40 --profile     --log-transform --voxel-cutoff 1.0     --output mcmc_sample_40_steps.hdf5 --n-samples 350000 --batch-size 50000


uv run python scripts/run_mcmc.py  \
    --cinn-ckpt /pscratch/sd/p/pmtuan/caloinn/caloinn_lightning/53921130/checkpoints/epoch=491-val_loss=895.95.ckpt   \
    --clf-ckpt /pscratch/sd/p/pmtuan/caloxtreme_clf/artifacts/best-54873288-val_loss=0.609829-epoch-epoch=999.ckpt     \
    --calibrator temperature_calibration.json     \
    --cinn-config params/pions_odd_sharded.yaml     \
    --n-steps 30 --profile     --log-transform --voxel-cutoff 1.0     --output mcmc_sample_30_steps.hdf5 --n-samples 350000 --batch-size 50000

uv run python scripts/run_mcmc.py  \
    --cinn-ckpt /pscratch/sd/p/pmtuan/caloinn/caloinn_lightning/53921130/checkpoints/epoch=491-val_loss=895.95.ckpt   \
    --clf-ckpt /pscratch/sd/p/pmtuan/caloxtreme_clf/artifacts/best-54873288-val_loss=0.609829-epoch-epoch=999.ckpt     \
    --calibrator temperature_calibration.json     \
    --cinn-config params/pions_odd_sharded.yaml     \
    --n-steps 20 --profile     --log-transform --voxel-cutoff 1.0     --output mcmc_sample_20_steps.hdf5 --n-samples 350000 --batch-size 50000

uv run python scripts/run_mcmc.py  \
    --cinn-ckpt /pscratch/sd/p/pmtuan/caloinn/caloinn_lightning/53921130/checkpoints/epoch=491-val_loss=895.95.ckpt   \
    --clf-ckpt /pscratch/sd/p/pmtuan/caloxtreme_clf/artifacts/best-54873288-val_loss=0.609829-epoch-epoch=999.ckpt     \
    --calibrator temperature_calibration.json     \
    --cinn-config params/pions_odd_sharded.yaml     \
    --n-steps 10 --profile     --log-transform --voxel-cutoff 1.0     --output mcmc_sample_10_steps.hdf5 --n-samples 350000 --batch-size 50000

uv run python scripts/run_mcmc.py  \
    --cinn-ckpt /pscratch/sd/p/pmtuan/caloinn/caloinn_lightning/53921130/checkpoints/epoch=491-val_loss=895.95.ckpt   \
    --clf-ckpt /pscratch/sd/p/pmtuan/caloxtreme_clf/artifacts/best-54873288-val_loss=0.609829-epoch-epoch=999.ckpt     \
    --calibrator temperature_calibration.json     \
    --cinn-config params/pions_odd_sharded.yaml     \
    --n-steps 60 --profile     --log-transform --voxel-cutoff 1.0     --output mcmc_sample_60_steps.hdf5 --n-samples 350000 --batch-size 50000

uv run python scripts/run_mcmc.py  \
    --cinn-ckpt /pscratch/sd/p/pmtuan/caloinn/caloinn_lightning/53921130/checkpoints/epoch=491-val_loss=895.95.ckpt   \
    --clf-ckpt /pscratch/sd/p/pmtuan/caloxtreme_clf/artifacts/best-54873288-val_loss=0.609829-epoch-epoch=999.ckpt     \
    --calibrator temperature_calibration.json     \
    --cinn-config params/pions_odd_sharded.yaml     \
    --n-steps 50 --profile     --log-transform --voxel-cutoff 1.0     --output mcmc_sample_50_steps.hdf5 --n-samples 350000 --batch-size 50000