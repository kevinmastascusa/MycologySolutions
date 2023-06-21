def parse_scientific_name(name):
    """
    Split a scientific name into its genus and species.
    """
    parts = name.split(' ')
    if len(parts) != 2:
        raise ValueError("Invalid scientific name. A valid name should have two parts: genus and species.")
    genus, species = parts
    return genus, species


genus, species = parse_scientific_name("Agaricus bisporus")
print(f"Genus: {genus}, Species: {species}")


#Genus: Agaricus, Species: bisporus
