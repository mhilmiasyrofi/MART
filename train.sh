# declare -a adv=("autoattack" "autopgd" "bim" "cw" "deepfool" "fgsm" "newtonfool" "pgd" "pixelattack" "spatialtransformation" "squareattack")

declare -a adv=("autoattack" "autopgd" "bim" "cw" "deepfool" "fgsm")
# declare -a adv=("newtonfool" "pgd" "pixelattack" "spatialtransformation" "squareattack")

# Iterate the string array using for loop
for a in ${adv[@]}; do
    python adv_train_resnet.py \
        --attack $a
done

# python adv_train_resnet.py \
#         --attack autoattack