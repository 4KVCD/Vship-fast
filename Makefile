current_dir := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

PREFIX ?= /usr/local
DESTDIR ?=

HIPARCH := gfx1201,gfx1200,gfx1100,gfx1101,gfx1102,gfx1103,gfx1151,gfx1012,gfx1030,gfx1031,gfx1032,gfx906,gfx801,gfx802,gfx803

ifeq ($(OS),Windows_NT)
	vulkanLink := -I "${VULKAN_SDK}\Include" -L "${VULKAN_SDK}\Lib" -lvulkan-1
    dllend := .dll
	exeend := .exe
    fpiccuda :=
    fpic :=
    plugin_install_path := $(APPDATA)\VapourSynth\plugins64
    exe_install_path := $(ProgramFiles)\FFVship.exe
    ffvshiplibheader :=  -lffms2 -llibvship
	ffvshipincludeheader := -I include
	fatbincompressamd := 
	fatbincompresscuda := 
else
	ifeq ($(VULKAN_SDK), )
		vulkanLink := -lvulkan
	else
		vulkanLink := -I "${VULKAN_SDK}/include" -L "${VULKAN_SDK}/lib" -lvulkan
	endif
    dllend := .so
	exeend :=
    fpiccuda := -Xcompiler -fPIC
    fpic := -fPIC
    plugin_install_path := $(DESTDIR)$(PREFIX)/lib/vapoursynth
	lib_install_path := $(DESTDIR)$(PREFIX)/lib
    exe_install_path := $(DESTDIR)$(PREFIX)/bin
	header_install_path := $(DESTDIR)$(PREFIX)/include
    ffvshiplibheader := $(shell pkg-config --libs ffms2) -lvship
	ffvshipincludeheader := $(shell pkg-config --cflags-only-I ffms2 libavutil)
	fatbincompressamd := 
	fatbincompresscuda := 
endif

.FORCE:

buildFFVSHIP: src/FFVship.cpp .FORCE
	clang++ src/FFVship.cpp -g -std=c++17 -Wall -O3 $(ffvshipincludeheader) -L ./ $(ffvshiplibheader) -o FFVship$(exeend)

build: src/VshipLib.cpp .FORCE
	hipcc src/VshipLib.cpp -g -std=c++17 -Wall -Wno-ignored-attributes -I "$(current_dir)include" --offload-arch=native -shared $(fpic) -o "$(current_dir)libvship$(dllend)"

buildcuda: src/VshipLib.cpp .FORCE
	nvcc -x cu src/VshipLib.cpp -g -std=c++17 -I "$(current_dir)include" -arch=native -shared $(fpiccuda) -o "$(current_dir)libvship$(dllend)"

buildcudaall: src/VshipLib.cpp .FORCE
	nvcc -x cu src/VshipLib.cpp -g -std=c++17 -I "$(current_dir)include" $(fatbincompresscuda) -arch=all -shared $(fpiccuda) -o "$(current_dir)libvship$(dllend)"

buildall: src/VshipLib.cpp .FORCE
	hipcc src/VshipLib.cpp -g -std=c++17 -Wall -Wno-ignored-attributes -I "$(current_dir)include" $(fatbincompressamd) --offload-arch=$(HIPARCH) -shared $(fpic) -o "$(current_dir)libvship$(dllend)"

shaderBuild: .FORCE
	slangc src/Vulkan/util/shaders/Planed.slang -O2 -target spirv -profile spirv_1_3 -emit-spirv-directly -fvk-use-entrypoint-name -entry StrideEliminator -entry StrideAdder -o libvshipSpvShaders/Planed.spv
	slangc src/Vulkan/util/shaders/kernelTest.slang -O2 -target spirv -profile spirv_1_3 -emit-spirv-directly -fvk-use-entrypoint-name -entry main -o libvshipSpvShaders/kernelTest.spv
	slangc src/Vulkan/util/shaders/basicOperations.slang -O2 -target spirv -profile spirv_1_3 -emit-spirv-directly -fvk-use-entrypoint-name -entry multArray -entry subArray -entry memSet -o libvshipSpvShaders/basicOperations.spv
	slangc src/Vulkan/gpuColorToLinear/shaders/anyDepthToFloat.slang -O2 -target spirv -profile spirv_1_3 -emit-spirv-directly -fvk-use-entrypoint-name -entry convertToFloatPlane -o libvshipSpvShaders/anyDepthToFloat.spv
	slangc src/Vulkan/gpuColorToLinear/shaders/chromaUpsample.slang -O2 -target spirv -profile spirv_1_3 -emit-spirv-directly -fvk-use-entrypoint-name -entry bicubicHorizontalCenterUpscaleX2 -entry bicubicHorizontalLeftUpscaleX2 -entry bicubicHorizontalCenterUpscaleX4 -entry bicubicHorizontalLeftUpscaleX4 -entry bicubicVerticalCenterUpscaleX2 -entry bicubicVerticalTopUpscaleX2 -o libvshipSpvShaders/chromaUpsample.spv
	slangc src/Vulkan/gpuColorToLinear/shaders/resize.slang -O2 -target spirv -profile spirv_1_3 -emit-spirv-directly -fvk-use-entrypoint-name -entry horizontalResizeTranspose -o libvshipSpvShaders/resize.spv
	slangc src/Vulkan/gpuColorToLinear/shaders/primaries.slang -O2 -target spirv -profile spirv_1_3 -emit-spirv-directly -fvk-use-entrypoint-name -entry primariesToPrimaries -o libvshipSpvShaders/primaries.spv
	slangc src/Vulkan/gpuColorToLinear/shaders/YUVToLinRGB.slang -O2 -target spirv -profile spirv_1_3 -emit-spirv-directly -fvk-use-entrypoint-name -entry YUVToLinRGB -o libvshipSpvShaders/YUVToLinRGB.spv
	slangc src/Vulkan/ssimu2/shaders/downsampleSSIMU2.slang -O2 -target spirv -profile spirv_1_3 -emit-spirv-directly -fvk-use-entrypoint-name -entry downsample -o libvshipSpvShaders/downsampleSSIMU2.spv
	slangc src/Vulkan/ssimu2/shaders/makeXYBSSIMU2.slang -O2 -target spirv -profile spirv_1_3 -emit-spirv-directly -fvk-use-entrypoint-name -entry rgb_to_positive_xyb -o libvshipSpvShaders/makeXYBSSIMU2.spv
	slangc src/Vulkan/ssimu2/shaders/sumreduceSSIMU2.slang -O2 -target spirv -profile spirv_1_3 -emit-spirv-directly -fvk-use-entrypoint-name -entry sumreduce -o libvshipSpvShaders/sumreduceSSIMU2.spv
	slangc src/Vulkan/ssimu2/shaders/scoreSSIMU2.slang -O2 -target spirv -profile spirv_1_3 -emit-spirv-directly -fvk-use-entrypoint-name -entry planescale_map -o libvshipSpvShaders/scoreSSIMU2.spv
	slangc src/Vulkan/butter/shaders/gaussianBlur.slang -O2 -target spirv -profile spirv_1_3 -emit-spirv-directly -fvk-use-entrypoint-name -entry TiledGaussianBlur -entry verticalBlur -entry horizontalBlur -o libvshipSpvShaders/gaussianBlurButter.spv
	slangc src/Vulkan/butter/shaders/downupsample.slang -O2 -target spirv -profile spirv_1_3 -emit-spirv-directly -fvk-use-entrypoint-name -entry addsupersample2X -entry downsample -o libvshipSpvShaders/downupsampleButter.spv
	slangc src/Vulkan/butter/shaders/combineMasks.slang -O2 -target spirv -profile spirv_1_3 -emit-spirv-directly -fvk-use-entrypoint-name -entry computeDiffmap -o libvshipSpvShaders/combineMasksButter.spv
	slangc src/Vulkan/butter/shaders/maltaDiff.slang -O2 -target spirv -profile spirv_1_3 -emit-spirv-directly -fvk-use-entrypoint-name -entry MaltaDiffMapLF -entry MaltaDiffMap -o libvshipSpvShaders/maltaDiffButter.spv
	slangc src/Vulkan/butter/shaders/colors.slang -O2 -target spirv -profile spirv_1_3 -emit-spirv-directly -fvk-use-entrypoint-name -entry opsinDynamicsImage -o libvshipSpvShaders/colorsButter.spv
	slangc src/Vulkan/butter/shaders/simplerDiff.slang -O2 -target spirv -profile spirv_1_3 -emit-spirv-directly -fvk-use-entrypoint-name -entry samenoisediff -entry diffclamp -entry L2diff -entry L2AsymDiff -o libvshipSpvShaders/simplerDiffButter.spv
	slangc src/Vulkan/butter/shaders/maskPsycho.slang -O2 -target spirv -profile spirv_1_3 -emit-spirv-directly -fvk-use-entrypoint-name -entry diffPrecompute -entry maskinit -entry fuzzyerrosion -o libvshipSpvShaders/maskPsychoButter.spv
	slangc src/Vulkan/butter/shaders/diffnorms.slang -O2 -target spirv -profile spirv_1_3 -emit-spirv-directly -fvk-use-entrypoint-name -entry sumreduce -entry sumreducenorm -o libvshipSpvShaders/diffnormsButter.spv
	slangc src/Vulkan/butter/shaders/separateFrequencies.slang -O2 -target spirv -profile spirv_1_3 -emit-spirv-directly -fvk-use-entrypoint-name -entry subarray_removerangearound0 -entry removerangearound0 -entry subarray_amplifyrangearound0 -entry supressXbyY -entry separateHf_Uhf -entry XybLowFreqToVals -o libvshipSpvShaders/separateFrequenciesButter.spv
	slangc src/Vulkan/cvvdp/shaders/distmap_specifics.slang -O2 -target spirv -profile spirv_1_3 -emit-spirv-directly -fvk-use-entrypoint-name -entry mergeChromaNorm -entry JODIZE -o libvshipSpvShaders/distmap_specificsCVVDP.spv
	slangc src/Vulkan/cvvdp/shaders/lpyr.slang -O2 -target spirv -profile spirv_1_3 -emit-spirv-directly -fvk-use-entrypoint-name -entry gaussPyrReduce -entry gaussPyrExpand -entry baseBandPyrRefine -o libvshipSpvShaders/lpyrCVVDP.spv
	slangc src/Vulkan/cvvdp/shaders/pooling.slang -O2 -target spirv -profile spirv_1_3 -emit-spirv-directly -fvk-use-entrypoint-name -entry reduceSum -o libvshipSpvShaders/poolingCVVDP.spv
	slangc src/Vulkan/cvvdp/shaders/colors.slang -O2 -target spirv -profile spirv_1_3 -emit-spirv-directly -fvk-use-entrypoint-name -entry XYZ_to_dkl -entry displayEncodePQ -entry displayEncodeBT709 -o libvshipSpvShaders/colorsCVVDP.spv
	slangc src/Vulkan/cvvdp/shaders/maskingModel.slang -O2 -target spirv -profile spirv_1_3 -emit-spirv-directly -fvk-use-entrypoint-name -entry preGaussianPreCompute -entry computeD -entry computeD_baseband -o libvshipSpvShaders/maskingModelCVVDP.spv
	slangc src/Vulkan/cvvdp/shaders/temporalFilter.slang -O2 -target spirv -profile spirv_1_3 -emit-spirv-directly -fvk-use-entrypoint-name -entry temporalConvolutionKernel_d -o libvshipSpvShaders/temporalFilterCVVDP.spv
	slangc src/Vulkan/cvvdp/shaders/resize.slang -O2 -target spirv -profile spirv_1_3 -emit-spirv-directly -fvk-use-entrypoint-name -entry horizontalResizeTranspose -o libvshipSpvShaders/resizeCVVDP.spv

shaderEmbedder: .FORCE
	clang++ src/Vulkan/spvFileToCppHeader.cpp -std=c++17 -O2 -o shaderEmbedder
	./shaderEmbedder libvshipSpvShaders include/libvshipSpvShaders.hpp

buildVulkan: shaderEmbedder src/VshipLib.cpp .FORCE
	clang++ src/VshipLib.cpp -DVULKANBUILD -DNDEBUG -std=c++17 -O2 -Wall -Wno-ignored-attributes -Wno-unused-variable -Wno-nullability-completeness -Wno-unused-private-field -I "$(current_dir)include" $(vulkanLink) $(fpic) -shared -o "$(current_dir)libvship$(dllend)"

debugVulkan: shaderEmbedder src/VshipLib.cpp .FORCE
	clang++ src/VshipLib.cpp -DVULKANBUILD -g -std=c++17 -O2 -Wall -Wno-ignored-attributes -Wno-unused-variable -Wno-nullability-completeness -Wno-unused-private-field -I "$(current_dir)include" $(vulkanLink) $(fpic) -shared -o "$(current_dir)libvship$(dllend)"


ifeq ($(OS),Windows_NT)
install:
	if exist "$(current_dir)libvship$(dllend)" copy "$(current_dir)libvship$(dllend)" "$(plugin_install_path)"
else
install:
	@if [ -f "$(current_dir)libvship$(dllend)" ]; then \
		install -d "$(plugin_install_path)"; \
		install -d "$(header_install_path)"; \
		install -m755 "$(current_dir)libvship$(dllend)" "$(lib_install_path)/libvship$(dllend)"; \
		ln -sf "../libvship$(dllend)" "$(plugin_install_path)/libvship$(dllend)"; \
		install -m755 "$(current_dir)src/VshipAPI.h" "$(header_install_path)/VshipAPI.h"; \
		install -m755 "$(current_dir)src/VshipColor.h" "$(header_install_path)/VshipColor.h"; \
	fi
	@if [ -f "FFVship" ]; then \
		install -d "$(exe_install_path)"; \
		install -m755 FFVship "$(exe_install_path)/FFVship"; \
	fi
uninstall:
	rm -f "$(plugin_install_path)/libvship$(dllend)" "$(lib_install_path)/libvship$(dllend)" "$(header_install_path)/VshipAPI.h" "$(header_install_path)/VshipColor.h" "$(exe_install_path)/FFVship"
uninstallOld:
	rm -f "$(plugin_install_path)/vship$(dllend)" "$(lib_install_path)/vship$(dllend)" "$(header_install_path)/VshipAPI.h" "$(exe_install_path)/FFVship"
endif

test: .FORCE build
	vspipe ./test/vsscript.vpy .
