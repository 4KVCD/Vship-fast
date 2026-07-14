### Vulkan Vship is slower

Yes! Vulkan Vship is currently not as optimized as the original due to lacking experience in Vulkan from my part. It mainly represents today a portability option. There may be some low hanging fruits remaining to be collected in the codebase if you wish to try it!

### Vulkan Vship on Adreno mobile GPU on Termux

2 users were able to make it work, however, it has been found that you need to install `mesa-vulkan-icd-freedreno` as a vulkan driver and not `vulkan-loader-android`. Otherwise you may get crashes on trying to run vship, failing to load some vulkan shaders.
