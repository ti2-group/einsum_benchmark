import pooch


def get_file_paths():
    POOCH = pooch.create(
        path=pooch.os_cache("einsum_benchmark"),
        base_url="doi:10.5281/zenodo.11477304",
        registry=None,
    )

    # Automatically populate the registry
    POOCH.load_registry_from_doi()

    # Fetch one of the files in the repository
    fname = POOCH.fetch(
        "instances.zip",
        progressbar=True,
        processor=pooch.Unzip(extract_dir="instances"),
    )

    # print(fname)
    sorted_fname = sorted(fname)
    return sorted_fname
