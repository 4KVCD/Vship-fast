import vapoursynth as vs
core = vs.core
#To print score, you need to use Python and not vspipe
#To output distmap, you need to remove prints, use vspipe and not python

# Load reference and distorted clips
ref = core.bs.VideoSource("../VshipUsageScript/sample.y4m")
dist = core.bs.VideoSource("../VshipUsageScript/testscores.mkv")

#==================SSIMULACRA2=====================

#numStream controls the performance-to-VRAM trade-off
ssimu2Clip = ref.vship.SSIMULACRA2(dist, numStream = 4)

# Extract scores from frame properties
ssimu2Scores = [frame.props["_SSIMULACRA2"] for frame in ssimu2Clip.frames()]

# Print average score
print(f"Average SSIMULACRA2 score: {sum(ssimu2Scores) / len(ssimu2Scores)}")

#==================Butteraugli=====================

# Calculate Butteraugli scores
# distmap controls whether to return a visual distortion map or the reference clip
# intensity_multiplier controls sensitivity
# qnorm controls which Norm to calculate and return in _BUTTERAUGLI_QNorm
butterClip = ref.vship.BUTTERAUGLI(dist, distmap = 0, numStream = 4, qnorm = 5)

butterScores = [[frame.props["_BUTTERAUGLI_3Norm"], frame.props["_BUTTERAUGLI_INFNorm"], frame.props["_BUTTERAUGLI_QNorm"]] for frame in butterClip.frames()]

# Output grayscale visualation of distortion for visual analysis for example with vs-preview
ref.vship.BUTTERAUGLI(dist, distmap = 1, numStream = 4).set_output()


#=====================CVVDP========================

# Calculate CVVDP scores
# distmap controls whether to return a visual distortion map or the reference clip
# model_name controls which Display Model to use
# resizeToDisplay conrols whether or not to resize the reference and distorted inputs to the Display Model resolution
cvvdpClip = ref.vship.CVVDP(dist, distmap = 0, model_name = "standard_fhd", resizeToDisplay = 0)

# Extract scores from frame properties

#wrong if temporal behavior is enabled since this method may not process frames in order
#scores = [frame.props["_CVVDP"] for frame in result.frames()]

#correct is:
cvvdpScores = []
for i in range(len(cvvdpClip)):
  cvvdpScores.append(cvvdpClip.get_frame(i).props["_CVVDP"])

# Only use the last score of CVVDP. (it takes into account every frame that it has seen up to now)
#it is different because it is an actually temporal metric unlike others
print(f"CVVDP Video Score: {cvvdpScores[-1]}")


# Output grayscale visualation of distortion for visual analysis
ref.vship.CVVDP(dist, distmap = 1).set_output()
