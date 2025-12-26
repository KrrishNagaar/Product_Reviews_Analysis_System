# Matplotlib for plotting charts
import matplotlib.pyplot as plt

# Pandas for counting and grouping review labels
import pandas as pd

# Use a clean, readable plotting style
plt.style.use("seaborn-v0_8-whitegrid")


# ===================== REVIEW CLASSIFICATION SUMMARY CHART =====================
def plot_review_summary(results):
    # If there are no results, do nothing
    if not results:
        return

    # Close any previously open figures to avoid overlap
    plt.close("all")

    # Normalize different prediction labels into fixed categories
    def normalize_label(label):
        label = label.lower()
        if "fake" in label:
            return "Fake"
        if "suspicious" in label:
            return "Suspicious"
        if "genuine" in label:
            return "Genuine"
        return "Neutral"

    # Convert model predictions into normalized labels
    labels = [normalize_label(r["prediction"]) for r in results]

    # Count how many reviews fall into each category
    counts = pd.Series(labels).value_counts()

    # Ensure consistent order of categories in the chart
    counts = counts.reindex(
        ["Fake", "Suspicious", "Genuine", "Neutral"],
        fill_value=0
    )

    # Create bar chart for review classification summary
    ax = counts.plot(
        kind="bar",
        title="Review Classification Summary",
        color=["#3D5A80", "#81B29A", "#F2CC8F", "#E07A5F"],
    )

    # Label the y-axis
    ax.set_ylabel("Number of Reviews", color="#2F3E46")

    # Set axis text colors for better readability
    plt.xticks(color="#2F3E46")
    plt.yticks(color="#2F3E46")

    # Adjust layout so labels do not overlap
    plt.tight_layout()

    # Add horizontal grid lines for easier comparison
    plt.grid(axis="y", color="#DADADA", linewidth=0.8)

    # Display the chart
    plt.show()


# ===================== AVERAGE CONFIDENCE CHART =====================
def plot_avg_confidence(results):
    # Dictionary to group confidence scores by review type
    groups = {
        "Likely Fake": [],
        "Suspicious": [],
        "Genuine": [],
        "Neutral": []
    }

    # Normalize prediction labels to standard categories
    def normalize_label(label):
        label = label.lower()
        if "fake" in label:
            return "Fake"
        if "suspicious" in label:
            return "Suspicious"
        if "genuine" in label:
            return "Genuine"
        return "Neutral"
        
    # Group confidence values based on prediction type
    for r in results:
        label = normalize_label(r.get("prediction", ""))
        conf = r.get("confidence", 0)
        groups[label].append(conf)

    labels = []
    avgs = []

    # Calculate average confidence for each review category
    for k, v in groups.items():
        if v:
            labels.append(k)
            avgs.append(sum(v) / len(v))

    # Create bar chart for average confidence
    plt.figure(figsize=(6, 4))
    plt.bar(
        labels,
        avgs,
        color=["#3D5A80", "#81B29A", "#F2CC8F", "#E07A5F"],
        edgecolor="#2F3E46",
        linewidth=1
    )

    # Label axes and title
    plt.ylabel("Average Confidence (%)", color="#2F3E46")
    plt.title("Average Confidence by Review Type", color="#2F3E46")

    # Set axis text colors
    plt.xticks(color="#2F3E46")
    plt.yticks(color="#2F3E46")

    # Adjust spacing for clean layout
    plt.tight_layout()

    # Add horizontal grid lines
    plt.grid(axis="y", color="#DADADA", linewidth=0.8)

    # Display the chart
    plt.show()
