#!/usr/bin/env bash
# Generate a synthetic podcast-shaped test video.
#
# This exists so Phase 1 and 2 can be tested without a real podcast. It mimics
# the SHAPE of a two-person podcast: 1920x1080 landscape, 29.97 fps, stereo AAC,
# two "speaker" regions side by side, and a burned-in timecode so you can verify
# by eye that clip timestamps land where they should.
#
# It contains no real speech. Transcription (Phase 3) and clip discovery
# (Phase 6) need a genuine podcast with dialogue — this fixture cannot test them.
#
# Usage:  ./scripts/make_test_video.sh [seconds] [output_path]

set -euo pipefail

DURATION="${1:-180}"
OUT="${2:-data/uploads/test_podcast.mp4}"
FFMPEG="${FFMPEG_BIN:-ffmpeg}"

mkdir -p "$(dirname "$OUT")"

echo "Generating ${DURATION}s synthetic podcast -> ${OUT}"

# Video: two coloured panels side by side, standing in for two speakers,
#        plus a large timecode overlay for verifying cut accuracy.
# Audio: two alternating tones, panned left and right, approximating a
#        two-speaker conversation with pauses between turns.
"$FFMPEG" -y -v error \
  -f lavfi -i "color=c=0x1b2838:s=1920x1080:r=30000/1001:d=${DURATION}" \
  -f lavfi -i "sine=frequency=180:duration=${DURATION}" \
  -f lavfi -i "sine=frequency=320:duration=${DURATION}" \
  -filter_complex "\
    [0:v]drawbox=x=140:y=240:w=740:h=600:color=0x2d4a6b@1:t=fill,\
         drawbox=x=1040:y=240:w=740:h=600:color=0x6b4a2d@1:t=fill,\
         drawtext=text='SPEAKER A':x=380:y=880:fontsize=44:fontcolor=white,\
         drawtext=text='SPEAKER B':x=1280:y=880:fontsize=44:fontcolor=white,\
         drawtext=text='%{pts\\:hms}':x=(w-tw)/2:y=80:fontsize=90:fontcolor=white:box=1:boxcolor=black@0.5:boxborderw=14\
    [v];\
    [1:a]volume='if(lt(mod(t,16),7),0.5,0)':eval=frame[a1];\
    [2:a]volume='if(gt(mod(t,16),8)*lt(mod(t,16),15),0.5,0)':eval=frame[a2];\
    [a1][a2]amix=inputs=2:duration=first[a]" \
  -map "[v]" -map "[a]" \
  -c:v libx264 -preset veryfast -crf 23 -pix_fmt yuv420p \
  -c:a aac -b:a 128k -ar 48000 -ac 2 \
  "$OUT"

echo "Done: ${OUT}"
"${FFPROBE_BIN:-ffprobe}" -v error -show_entries \
  format=duration,size:stream=codec_name,width,height,avg_frame_rate \
  -of default=noprint_wrappers=1 "$OUT"
