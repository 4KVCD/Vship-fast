current_dir := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

PREFIX ?= /usr/local
DESTDIR ?=
CXX ?= clang++
BACKEND ?= HIP
#can be Cuda and Vulkan
CXXFLAGS ?=
LDFLAGS ?=
ARCH ?= native
#can be native or all
DEBUG ?= 0

HIPARCH := gfx1201,gfx1200,gfx1100,gfx1101,gfx1102,gfx1103,gfx1151,gfx1012,gfx1030,gfx1031,gfx1032,gfx906,gfx801,gfx802,gfx803

VSHIP_VERSION_MAJOR=5
VSHIP_VERSION_MINOR=0
VSHIP_VERSION_MINORMINOR=2

CXXFLAGS += -DVSHIP_VERSION_MAJOR=$(VSHIP_VERSION_MAJOR) -DVSHIP_VERSION_MINOR=$(VSHIP_VERSION_MINOR) -DVSHIP_VERSION_MINORMINOR=$(VSHIP_VERSION_MINORMINOR)
VSHIP_CXXFLAGS += -std=c++17 -I "$(current_dir)include" -shared
ifeq ($(DEBUG), 1)
	VSHIP_CXXFLAGS += -g -O2
else
	VSHIP_CXXFLAGS += -DNDEBUG -O3
endif
ifeq ($(MITIGATE_MALLOC_ASYNC), on)
	VSHIP_CXXFLAGS += -DMITIGATE_MALLOC_ASYNC
endif

ifeq ($(BACKEND), HIP)
	VSHIP_CXX = hipcc
	VSHIP_CXXFLAGS += -Wno-ignored-attributes
	ifeq ($(ARCH), native)
		VSHIP_CXXFLAGS += --offload-arch=native
	else
		VSHIP_CXXFLAGS += --offload-arch=$(HIPARCH)
	endif
	ifeq ($(OS),Windows_NT)

	else
		VSHIP_CXXFLAGS += -fPIC
	endif
else ifeq ($(BACKEND), Cuda)
	VSHIP_CXX = nvcc -x cu
	ifeq ($(ARCH), native)
		VSHIP_CXXFLAGS += -arch=native
	else
		VSHIP_CXXFLAGS += -arch=all
	endif
	ifeq ($(OS),Windows_NT)

	else
		VSHIP_CXXFLAGS += -Xcompiler -fPIC
	endif
else ifeq ($(BACKEND), Vulkan)
	VSHIP_CXX = $(CXX)
	VSHIP_CXXFLAGS += -DVULKANBUILD -Wno-ignored-attributes -Wno-unused-variable -Wno-nullability-completeness -Wno-unused-private-field
	ifeq ($(OS),Windows_NT)
		VSHIP_LDFLAGS += -I "${VULKAN_SDK}\Include" -L "${VULKAN_SDK}\Lib" -lvulkan-1
	else
		ifeq ($(VULKAN_SDK), )
			VSHIP_LDFLAGS += -lvulkan
		else
			LDFLAGS += -I "${VULKAN_SDK}/include" -L "${VULKAN_SDK}/lib" -lvulkan
		endif
		VSHIP_CXXFLAGS += -fPIC
	endif
else
	ERROR := $(error $(BACKEND) is not a valid Backend, choose HIP, Cuda or Vulkan)
endif

FFVSHIP_CXXFLAGS += -std=c++17
ifeq ($(DEBUG), 1)
	FFVSHIP_CXXFLAGS += -g -O2
else
	FFVSHIP_CXXFLAGS += -O3
endif

ifeq ($(OS),Windows_NT)
	dllend := .dll
	exeend := .exe

	plugin_install_path := $(APPDATA)\VapourSynth\plugins64
	exe_install_path := $(ProgramFiles)\FFVship.exe

	FFVSHIP_LDFLAGS +=  -lffms2 -llibvship -L ./
	FFVSHIP_CXXFLAGS += -I include
else
	dllend := .so
	exeend :=

	plugin_install_path := $(DESTDIR)$(PREFIX)/lib/vapoursynth
	lib_install_path := $(DESTDIR)$(PREFIX)/lib
	exe_install_path := $(DESTDIR)$(PREFIX)/bin
	header_install_path := $(DESTDIR)$(PREFIX)/include
	pkgconfig_install_path := $(lib_install_path)/pkgconfig

	FFVSHIP_LDFLAGS += $(shell pkg-config --libs ffms2) -lvship -L ./
	FFVSHIP_CXXFLAGS += $(shell pkg-config --cflags-only-I ffms2 libavutil)

	#finding libcudart
	#CUDA_LIBPATH := $(abspath $(dir $(abspath $(lastword $(shell which nvcc))))../lib64)
endif

ifeq ($(OS),Windows_NT)
	OLDVulkanLink = -I "${VULKAN_SDK}\Include" -L "${VULKAN_SDK}\Lib" -lvulkan-1
else
	ifeq ($(VULKAN_SDK), )
		OLDVulkanLink = -lvulkan
	else
		OLDVulkanLink = -I "${VULKAN_SDK}/include" -L "${VULKAN_SDK}/lib" -lvulkan
	endif
endif

define PKG_CONFIG
prefix=$(DESTDIR)$(PREFIX)
exec_prefix=${prefix}
includedir=${prefix}/include
libdir=${exec_prefix}/lib

Name: libvship
Description: GPU Accelerated video quality metric library
Version: $(VSHIP_VERSION_MAJOR).$(VSHIP_VERSION_MINOR).$(VSHIP_VERSION_MINORMINOR)
Cflags: -I $(header_install_path)
Libs: -L $(lib_install_path) -lvship"
endef

.FORCE:

buildFFVSHIP: src/FFVship.cpp .FORCE
	$(CXX) src/FFVship.cpp $(CXXFLAGS) $(LDFLAGS) $(FFVSHIP_CXXFLAGS) $(FFVSHIP_LDFLAGS) -o FFVship$(exeend)

shaderEmbedder: .FORCE
ifeq ($(BACKEND),Vulkan)
	$(CXX) src/Vulkan/spvFileToCppHeader.cpp -std=c++17 -O2 -o shaderEmbedder
	./shaderEmbedder libvshipSpvShaders include/libvshipSpvShaders.hpp
endif

genpkgConfig: .FORCE
	$(file > libvship.pc,$(PKG_CONFIG))

build: shaderEmbedder src/VshipLib.cpp .FORCE
	$(VSHIP_CXX) src/VshipLib.cpp $(CXXFLAGS) $(LDFLAGS) $(VSHIP_CXXFLAGS) $(VSHIP_LDFLAGS) -o "$(current_dir)libvship$(dllend)"

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

ifeq ($(OS),Windows_NT)
install:
	if exist "$(current_dir)libvship$(dllend)" copy "$(current_dir)libvship$(dllend)" "$(plugin_install_path)"
else
install:
	@if [ -f "$(current_dir)libvship$(dllend)" ]; then \
		install -d "$(plugin_install_path)"; \
		install -d "$(header_install_path)"; \
		install -d "$(pkgconfig_install_path)"; \
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

buildcuda: src/VshipLib.cpp .FORCE
	echo DEPRECATED BUILDING SYSTEM USED
	nvcc -x cu src/VshipLib.cpp -g -std=c++17 $(CXXFLAGS) -I "$(current_dir)include" -arch=native -shared $(mitigateAsync) -Xcompiler -fPIC -o "$(current_dir)libvship$(dllend)"

buildcudaall: src/VshipLib.cpp .FORCE
	echo DEPRECATED BUILDING SYSTEM USED
	nvcc -x cu src/VshipLib.cpp -g -std=c++17 $(CXXFLAGS) -I "$(current_dir)include" $(fatbincompresscuda) -arch=all -shared $(mitigateAsync) -Xcompiler -fPIC -o "$(current_dir)libvship$(dllend)"

buildall: src/VshipLib.cpp .FORCE
	echo DEPRECATED BUILDING SYSTEM USED
	hipcc src/VshipLib.cpp -g -std=c++17 -Wall -Wno-ignored-attributes $(CXXFLAGS) -I "$(current_dir)include" $(fatbincompressamd) --offload-arch=$(HIPARCH) -shared $(mitigateAsync) -fPIC -o "$(current_dir)libvship$(dllend)"

buildVulkan: src/VshipLib.cpp .FORCE
	echo DEPRECATED BUILDING SYSTEM USED
	clang++ src/Vulkan/spvFileToCppHeader.cpp -std=c++17 -O2 -o shaderEmbedder
	./shaderEmbedder libvshipSpvShaders include/libvshipSpvShaders.hpp
	clang++ src/VshipLib.cpp -DVULKANBUILD -DNDEBUG $(CXXFLAGS) -std=c++17 -O2 -Wall -Wno-ignored-attributes -Wno-unused-variable -Wno-nullability-completeness -Wno-unused-private-field -I "$(current_dir)include" $(OLDVulkanLink) -fPIC -shared -o "$(current_dir)libvship$(dllend)"

debugVulkan: shaderEmbedder src/VshipLib.cpp .FORCE
	echo DEPRECATED BUILDING SYSTEM USED
	clang++ src/VshipLib.cpp -DVULKANBUILD -g $(CXXFLAGS) -std=c++17 -O2 -Wall -Wno-ignored-attributes -Wno-unused-variable -Wno-nullability-completeness -Wno-unused-private-field -I "$(current_dir)include" $(vulkanLink) -fPIC -shared -o "$(current_dir)libvship$(dllend)"
