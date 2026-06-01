import argparse

from PIL import Image, ImageDraw, ImageFont


def label(path, text):
    image = Image.open(path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.rectangle((24, 24, 160, 58), fill="white")
    draw.text((36, 36), text, fill="black", font=font)
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", required=True)
    parser.add_argument("--after", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    before = label(args.before, "Before")
    after = label(args.after, "After")
    figure = Image.new("RGB", (before.width + after.width, max(before.height, after.height)), "white")
    figure.paste(before, (0, 0))
    figure.paste(after, (before.width, 0))
    figure.save(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
