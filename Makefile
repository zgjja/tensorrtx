.PHONY: clang-format-check run-clang-tidy

# format all C/C++/CUDA files
CLANG_FORMAT_FILES = \
	-name '*.c'   -o \
	-name '*.cc'  -o \
	-name '*.cpp' -o \
	-name '*.cxx' -o \
	-name '*.h'   -o \
	-name '*.hpp' -o \
	-name '*.hxx' -o \
	-name '*.inl' -o \
	-name '*.cuh' -o \
	-name '*.cuhpp' -o \
	-name '*.cu' -o \
	-false

# for now, only include subprojects in this repo by whitelist in the early stage of refactoring,
# maybe include them all in the future
EXCLUDE_DIRS = \
	-path './yolov9' -o -path './yolov9/*' -o \
	-path './build' -o -path './build/*' -o \
	-path './arcface' -o -path './arcface/*' -o \
	-path './centernet' -o -path './centernet/*' -o \
	-path './convnextv2' -o -path './convnextv2/*' -o \
	-path './crnn' -o -path './crnn/*' -o \
	-path './csrnet' -o -path './csrnet/*' -o \
	-path './dbnet' -o -path './dbnet/*' -o \
	-path './densenet' -o -path './densenet/*' -o \
	-path './detr' -o -path './detr/*' -o \
	-path './efficient_ad' -o -path './efficient_ad/*' -o \
	-path './efficientnet' -o -path './efficientnet/*' -o \
	-path './ghostnet' -o -path './ghostnet/*' -o \
	-path './hrnet' -o -path './hrnet/*' -o \
	-path './ibnnet' -o -path './ibnnet/*' -o \
	-path './inception' -o -path './inception/*' -o \
	-path './lprnet' -o -path './lprnet/*' -o \
	-path './mnasnet' -o -path './mnasnet/*' -o \
	-path './mobilenet' -o -path './mobilenet/*' -o \
	-path './models' -o -path './models/*' -o \
	-path './psenet' -o -path './psenet/*' -o \
	-path './rcnn' -o -path './rcnn/*' -o \
	-path './real-esrgan' -o -path './real-esrgan/*' -o \
	-path './refinedet' -o -path './refinedet/*' -o \
	-path './repvgg' -o -path './repvgg/*' -o \
	-path './resnet' -o -path './resnet/*' -o \
	-path './retinaface' -o -path './retinaface/*' -o \
	-path './retinafaceAntiCov' -o -path './retinafaceAntiCov/*' -o \
	-path './sam3' -o -path './sam3/*' -o \
	-path './scaled-yolov4' -o -path './scaled-yolov4/*' -o \
	-path './scripts' -o -path './scripts/*' -o \
	-path './senet' -o -path './senet/*' -o \
	-path './shufflenetv2' -o -path './shufflenetv2/*' -o \
	-path './squeezenet' -o -path './squeezenet/*' -o \
	-path './superpoint' -o -path './superpoint/*' -o \
	-path './swin-transformer' -o -path './swin-transformer/*' -o \
	-path './tsm' -o -path './tsm/*' -o \
	-path './tutorials' -o -path './tutorials/*' -o \
	-path './ufld' -o -path './ufld/*' -o \
	-path './unet' -o -path './unet/*' -o \
	-path './vgg' -o -path './vgg/*' -o \
	-path './vit' -o -path './vit/*' -o \
	-path './yolo11' -o -path './yolo11/*' -o \
	-path './yolo11_tripy' -o -path './yolo11_tripy/*' -o \
	-path './yolop' -o -path './yolop/*' -o \
	-path './yolov10' -o -path './yolov10/*' -o \
	-path './yolov12' -o -path './yolov12/*' -o \
	-path './yolov12-tubro' -o -path './yolov12-tubro/*' -o \
	-path './yolov13' -o -path './yolov13/*' -o \
	-path './yolov3' -o -path './yolov3/*' -o \
	-path './yolov3-spp' -o -path './yolov3-spp/*' -o \
	-path './yolov3-tiny' -o -path './yolov3-tiny/*' -o \
	-path './yolov4' -o -path './yolov4/*' -o \
	-path './yolov5' -o -path './yolov5/*' -o \
  -path './yolov5-lite' -o -path './yolov5-lite/*' -o \
  -path './yolov7' -o -path './yolov7/*' -o \
  -path './yolov8' -o -path './yolov8/*' -o \
  -false

run-clang-format:
	@echo "==> Running clang-format check"
	@find . \
		\( $(EXCLUDE_DIRS) \) -prune -o \
		-type f \( $(CLANG_FORMAT_FILES) \) \
		-print0 \
    | xargs -0 -r clang-format --dry-run --Werror

BUILD_DIR ?= build/Debug
CLANG_TIDY_DB ?= compile_commands.json
CLANG_TIDY_WARNINGS_AS_ERRORS ?= *
REPO_ABS_ROOT := $(abspath $(dir $(MAKEFILE_LIST)))

run-clang-tidy:
	@echo "==> Running run-clang-tidy"
	@echo "current dir: $(REPO_ABS_ROOT)"
	@run-clang-tidy-18 -quiet \
		-warnings-as-errors='$(CLANG_TIDY_WARNINGS_AS_ERRORS)' \
		-header-filter='^$(REPO_ABS_ROOT)/.*' \
		-p '$(BUILD_DIR)' \
		-j 5
