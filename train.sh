# declare -a adv=("autoattack" "autopgd" "bim" "cw" "deepfool" "fgsm" "newtonfool" "pgd" "pixelattack" "spatialtransformation" "squareattack")

declare -a adv=("autoattack" "autopgd" "bim" "cw" "deepfool" "fgsm")
# declare -a adv=("newtonfool" "pgd" "pixelattack" "spatialtransformation" "squareattack")

# Iterate the string array using for loop
# for a in ${adv[@]}; do
#     python adv_train_resnet.py \
#         --attack $a
# done

# python adv_train_resnet.py \
#         --attack autoattack

# python adv_train_resnet.py \
#         --attack all
        
python adv_train_resnet.py \
        --attack combine \
        --list autopgd_pixelattack
        
python adv_train_resnet.py \
        --attack combine \
        --list autopgd_pixelattack_spatialtransformation
        
python adv_train_resnet.py \
        --attack combine \
        --list autopgd_pixelattack_spatialtransformation_squareattack

python adv_train_resnet.py \
        --attack combine \
        --list newtonfool_spatialtransformation_pixelattack_autoattack_squareattack
        
python adv_train_resnet.py \
        --attack combine \
        --list autopgd_pixelattack_spatialtransformation_squareattack_fgsm_newtonfool
        
python adv_train_resnet.py \
        --attack combine \
        --list newtonfool_spatialtransformation_pixelattack_autopgd_squareattack_cw_pgd
        
python adv_train_resnet.py \
        --attack combine \
        --list cw_pixelattack_spatialtransformation_autopgd_squareattack_bim_fgsm_pgd
        
python adv_train_resnet.py \
        --attack combine \
        --list cw_pixelattack_spatialtransformation_autopgd_squareattack_bim_fgsm_pgd_newtonfool
        