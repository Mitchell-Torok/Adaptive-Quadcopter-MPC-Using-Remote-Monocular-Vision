#!/usr/bin/env bash
# Fetches the unmodified upstream ros2_orb_slam3 package (which bundles ORB-SLAM3 V1.0,
# its Thirdparty libraries and the ORB vocabulary) into src/ros2_orb_slam3 and overlays
# the interface files from this directory on top of it.
#
# Usage (from the repository root):   ./orbslam_interface/setup_orbslam.sh
set -euo pipefail

UPSTREAM_URL="https://github.com/Mechazo11/ros2_orb_slam3.git"
UPSTREAM_COMMIT="ee7d378"   # upstream revision the interface was developed against

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
DEST="$ROOT/src/ros2_orb_slam3"

if [ -d "$DEST" ]; then
  echo "$DEST already exists - remove it first if you want a fresh copy."
  exit 1
fi

echo ">> Cloning upstream ros2_orb_slam3 @ $UPSTREAM_COMMIT"
git clone --quiet "$UPSTREAM_URL" "$DEST"
git -C "$DEST" checkout --quiet "$UPSTREAM_COMMIT"
rm -rf "$DEST/.git"

echo ">> Removing upstream sample dataset and example configs that are not needed"
rm -rf "$DEST/TEST_DATASET" "$DEST/scripts" "$DEST/orb_slam3/config/Monocular" \
       "$DEST/orb_slam3/config/Monocular-Inertial" "$DEST/orb_slam3/config/Stereo" \
       "$DEST/orb_slam3/config/Stereo-Inertial" "$DEST/orb_slam3/config/RGB-D" \
       "$DEST/orb_slam3/config/RGB-D-Inertial"

echo ">> Applying the ORB-SLAM3 core patch (exposes the viewer image)"
patch -p1 -d "$DEST" < "$HERE/orb_slam3_viewer_image.patch"

echo ">> Overlaying the ROS 2 interface files"
cp    "$HERE/CMakeLists.txt" "$HERE/package.xml"        "$DEST/"
cp    "$HERE/src/common.cpp"                            "$DEST/src/common.cpp"
cp    "$HERE/include/ros2_orb_slam3/common.hpp"         "$DEST/include/ros2_orb_slam3/common.hpp"
mkdir -p "$DEST/orb_slam3/config/Monocular" "$DEST/scripts"
cp    "$HERE"/config/Monocular/*.yaml                   "$DEST/orb_slam3/config/Monocular/"
cp    "$HERE"/scripts/*.py                              "$DEST/scripts/"

echo ">> Done. Build with: colcon build --symlink-install"
