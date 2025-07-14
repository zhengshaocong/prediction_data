from matplotlib.font_manager import FontProperties
import os

def get_chinese_font():
    """
    获取可用的中文字体 FontProperties 对象，自动适配常见操作系统。
    """
    font_paths = [
        '/System/Library/Fonts/PingFang.ttc',  # macOS 苹方
        '/System/Library/Fonts/Hiragino Sans GB.ttc',  # macOS
        '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',  # Linux 文泉驿
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',  # Linux Noto
        'C:/Windows/Fonts/simhei.ttf',  # Windows 黑体
        'C:/Windows/Fonts/msyh.ttc',    # Windows 微软雅黑
    ]
    for path in font_paths:
        if os.path.exists(path):
            return FontProperties(fname=path)
    # 若未找到，返回 None，matplotlib 默认字体
    return None 