# declare -a train=("original" "autoattack" "autopgd" "bim" "cw" "deepfool" "fgsm" "newtonfool" "pgd" "pixelattack" "spatialtransformation" "squareattack")


declare -a train=("original" "autoattack" "autopgd" "bim" "cw" "deepfool")
# declare -a train=("fgsm" "newtonfool" "pgd" "pixelattack" "squareattack" "spatialtransformation")

declare -a test=("autoattack" "autopgd" "bim" "cw" "deepfool" "fgsm" "newtonfool" "pgd" "pixelattack" "spatialtransformation" "squareattack")


# Iterate the string array using for loop
for tr in ${train[@]}; do
    for ts in ${test[@]}; do
        python eval.py \
            --train-adversarial $tr \
            --test-adversarial $ts
    done
done

# python eval.py \
#     --train-adversarial squareattack \
#     --test-adversarial squareattack