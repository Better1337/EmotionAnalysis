import matplotlib.pyplot as plt

    # Plot the emotion pie chart
def plot_emotion_pie_chart(figure, canvas, percentages):
    figure.clear()
    ax = figure.add_subplot(111)

    # Extract labels and sizes
    labels = [item[0] for item in percentages]
    sizes = [item[1] for item in percentages]

    # Create the pie chart
    wedges, texts, autotexts = ax.pie(
        sizes, 
        labels=labels, 
        autopct='%1.1f%%', 
        startangle=140,  
        textprops={'fontsize': 10}, 
        wedgeprops={'edgecolor': 'black'}
    )

    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(7)

    # Equal aspect ratio ensures that pie is drawn as a circle
    ax.axis('equal')
    canvas.draw()



