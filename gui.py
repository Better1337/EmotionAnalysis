import matplotlib.pyplot as plt

def plot_emotion_pie_chart(figure, canvas, emotion_probabilities, threshold=0.05):
    figure.clear()
    ax = figure.add_subplot(111)

    filtered_emotions = {k: v for k, v in emotion_probabilities.items() if v >= threshold}
    labels = list(filtered_emotions.keys())
    sizes = list(filtered_emotions.values())

    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct='%1.1f%%', startangle=140,
        textprops={'fontsize': 10}, wedgeprops={'edgecolor': 'black'}
    )

    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(8)

    ax.axis('equal')
    canvas.draw()
