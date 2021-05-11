size="small"
dataset="mutual"
model_name="google/electra-small-discriminator"

python run_model_test.py  \
  --test  \
  --data_dir "mutual/"   \
  --checkpoint "epoch7_global-step56117" \
  --test_batch_size 4  \
  --model_name  "google/electra-small-discriminator"  \
  --output_dir "mutual-speaker_embeddings-electra-small-ckpts/epoch8_global-step_output" \
  --speaker_embeddings \
