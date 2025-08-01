import matplotlib.pyplot as plt


def save_text_as_image(text, save_path):
    plt.figure(figsize=(10, 5))
    plt.axis('off')
    plt.text(0.01, 0.99, text, fontsize=12, va='top', family='monospace')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()