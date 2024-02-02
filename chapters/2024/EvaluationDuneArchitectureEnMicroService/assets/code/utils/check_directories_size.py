import yaml

class services_size():

    def __init__(self):
        print("Info ", flush=True)


    def is_large_directory(self, repository, threshold_factor=2):
        # Getting the size of each directory
        contents = repository.get_contents("")
        directory_sizes = []

        for content in contents:
            if content.type == "dir" and "deliverable" not in content.name  and "doc" not in content.name and "front" not in content.name and "web" not in content.name:
                directory_size = self.calculate_directory_size(repository, content.path)
                directory_sizes.append((content.path, directory_size))

        # Find the average size of all directories
        average_size = sum(size for _, size in directory_sizes) / len(directory_sizes)

        # Find the directory with the maximum size
        largest_directory, largest_directory_size = max(directory_sizes, key=lambda x: x[1])

        # Check if the largest directory size is significantly larger than the average size
        if largest_directory_size > threshold_factor * average_size:
            return True, largest_directory, largest_directory_size, average_size
        else:
            return False, largest_directory, largest_directory_size, average_size

    def calculate_directory_size(self, repository, path):
        try:
            contents = repository.get_contents(path)
            directory_size = 0

            for content in contents:
                if content.type == "file":
                    directory_size += content.size

            return directory_size

        except Exception as e:
            print(f"Error calculating directory size: {e}")
            return 0
    

