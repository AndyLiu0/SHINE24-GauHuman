#SEQUENCES=("lan_images620_1300" "marc_images35000_36200" "olek_images0812" "vlad_images1011")
dataset="data/andy2"
python train.py -s $dataset --eval --exp_name andy/andy2 --motion_offset_flag --smpl_type smpl --actor_gender neutral --iterations 2000
