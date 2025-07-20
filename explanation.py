def generate_explanation(query, caption):
    return f"The image is relevant because it shows: '{caption}', which aligns with the query: '{query}'."

# Example usage
if __name__ == "__main__":
    q = "a man riding a horse"
    c = "A person on a brown horse galloping in a field"
    print(generate_explanation(q, c))
