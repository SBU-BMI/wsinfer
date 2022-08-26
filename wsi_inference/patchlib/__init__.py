# Create tissue mask and patch a whole slide image.
# Copyright (C) 2022  Mahmood Lab
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# Modified by Jakub Kaczmarzyk (@kaczmarj on GitHub)
# - add --patch_spacing command line arg to request a patch size at a particular
#   spacing. The patch coordinates are calculated at the base (highest) resolution.
# - format code with black

from .create_patches_fp import cli
