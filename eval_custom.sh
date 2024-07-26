#SEQUENCES=("lan_images620_1300" "marc_images35000_36200" "olek_images0812" "vlad_images1011")
#SEQUENCES=("lan_images620_1300" "marc_images35000_3620
dataset="data/andy2"
python render.py -m output/andy/andy2 --motion_offset_flag --smpl_type smpl --actor_gender neutral --iteration 2000
