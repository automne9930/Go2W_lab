# 资产配置（直接搬运了robot-lab的配置文件）

import os
import toml

##
# Configuration for different assets.
##

# 当前扩展包的根目录绝对路径（基于当前文件向上回退两层定位）
ISAACLAB_ASSETS_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
"""Path to the extension source directory."""

# 通过根目录定位到扩展包的data目录
ISAACLAB_ASSETS_DATA_DIR = os.path.join(ISAACLAB_ASSETS_EXT_DIR, "data")
"""Path to the extension data directory."""

# 通过根目录定位到扩展包的config目录
ISAACLAB_ASSETS_METADATA = toml.load(os.path.join(ISAACLAB_ASSETS_EXT_DIR, "config", "extension.toml"))
"""Extension metadata dictionary parsed from the extension.toml file."""

# 从配置中动态提取版本号
__version__ = ISAACLAB_ASSETS_METADATA["package"]["version"]