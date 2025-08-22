salloc \
    --job-name=brats \
    --partition=dgx-small \
    --account=ddt_acc23 \
    --time=3-00:00:00 \
    --nodes=1 \
    --ntasks-per-node=1 \
    --gres=gpu:a100:1 \
    --cpus-per-task=16 \
    --mem=40G