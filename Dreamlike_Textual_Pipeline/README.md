Unzip the folders

Use the command below after making changes in the locations to run the code.

accelerate launch --num_processes=1 --main_process_port=0  /mnt/local/swathi/diffusers/examples/textual_inversion/textual_inversion.py \
--pretrained_model_name_or_path=$MODEL_NAME \
--train_data_dir=$DATA_DIR \
--learnable_property="" \
--placeholder_token="<bajiface3>" \
--initializer_token="man" \
--resolution=512 \
--train_batch_size=1 \
--gradient_accumulation_steps=4 \
--max_train_steps=5000 \
--learning_rate=5.0e-04 \
--scale_lr \
--lr_scheduler="constant" \
--lr_warmup_steps=500 \
--output_dir="textual_inversion_bajirao_scene2" \
