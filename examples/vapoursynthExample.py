import vapoursynth as vs
core = vs.core

# Load reference and distorted clips
ref = core.bs.VideoSource("reference.mp4")
dist = core.bs.VideoSource("distorted.mp4")

#==================SSIMULACRA2=====================

#numStream controls the performance-to-VRAM trade-off
ssimu2Clip = ref.vship.SSIMULACRA2(dist, numStream = 4)

# Extract scores from frame properties
SSIMU2scores = [frame.props["_SSIMULACRA2"] for frame in ssimu2Clip.frames()]

# Print average score
print(f"Average SSIMULACRA2 score: {sum(scores) / len(scores)}")

#==================Butteraugli=====================

# Calculate Butteraugli scores
# distmap controls whether to return a visual distortion map or the reference clip
# intensity_multiplier controls sensitivity
# qnorm controls which Norm to calculate and return in _BUTTERAUGLI_QNorm
butterClip = ref.vship.BUTTERAUGLI(dist, distmap = 0, numStream = 4, qnorm = 5)

all_scores_Butteraugli = [frame.props["_BUTTERAUGLI_3Norm"],
               frame.props["_BUTTERAUGLI_INFNorm"],
               frame.props["_BUTTERAUGLI_QNorm"]]
              for frame in result.frames()]

# Output grayscale visualation of distortion for visual analysis for example with vs-preview
ref.vship.BUTTERAUGLI(dist, distmap = 1, numStream = 4).set_output()


#=====================CVVDP========================

# Calculate CVVDP scores
# distmap controls whether to return a visual distortion map or the reference clip
# model_name controls which Display Model to use
# resizeToDisplay conrols whether or not to resize the reference and distorted inputs to the Display Model resolution
CVVDPClip = ref.vship.CVVDP(dist, distmap = 0, model_name = "standard_fhd", resizeToDisplay = 0)

# Extract scores from frame properties

#wrong if temporal behavior is enabled since this method may not process frames in order
#scores = [frame.props["_CVVDP"] for frame in result.frames()]

#correct is:
scores = []
for i in range(len(result)):
  scores.append(result.get_frame(i).props["_CVVDP"])

# Only use the last score of CVVDP. (it takes into account every frame that it has seen up to now)
#it is different because it is an actually temporal metric unlike others
print(f"CVVDP Video Score: {scores[-1]}")


# Output grayscale visualation of distortion for visual analysis
ref.vship.CVVDP(dist, distmap = 1).set_output()
