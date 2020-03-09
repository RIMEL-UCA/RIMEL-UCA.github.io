package fr.unice.polytech.si5.rimel.eventprofile.downloader;

import fr.unice.polytech.si5.rimel.eventprofile.Main;
import org.eclipse.jgit.api.Git;
import org.eclipse.jgit.api.errors.GitAPIException;
import org.eclipse.jgit.lib.TextProgressMonitor;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

public class OptimizedDownloader {

	/**
	 * Open the repo, download it if it is not present into the workspace.
	 *
	 * @param repositoryURL the repo url
	 * @return the jgit repo
	 * @throws IOException
	 * @throws GitAPIException
	 */
	public Git openRepository(final String repositoryURL) throws IOException, GitAPIException, InterruptedException {
		// Setup the git repo
		String[] project = repositoryURL.split("/");
		String projectName = project[project.length-1];
		Path path = Paths.get(Main.WORKSPACE + projectName + "/.git");
		Git repo;
		if (!Files.exists(path)) {
			System.out.println("Cloning repo : " + repositoryURL);
			this.clone(repositoryURL, projectName);
			repo = Git.open(new File(Main.WORKSPACE + projectName + "/.git"));
		} else {
			System.out.println("Open repo : " + projectName);
			repo = Git.open(new File(Main.WORKSPACE + projectName + "/.git"));
		}
		return repo;
	}

	private void clone(final String repoUrl, final String projectName) throws IOException, InterruptedException {

		// Launch git clone command
		Process process = Runtime.getRuntime().exec("git clone --no-checkout " + repoUrl + " " + Main.WORKSPACE + projectName);
		StringBuilder output = new StringBuilder();
		BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));

		BufferedReader errorReader = new BufferedReader(new InputStreamReader(process.getErrorStream()));

		String line;
		while ((line = reader.readLine()) != null) {
			output.append(line + " ");
		}

		int exitVal = process.waitFor();
		if (exitVal == 0) {
			System.out.println("Git clone success.");
		} else {
			StringBuilder error = new StringBuilder();
			while ((line = errorReader.readLine()) != null) {
				error.append(line + " ");
			}
			System.err.println("Something went wrong in the git clone command");
			System.err.println(error.toString());
			System.err.println(exitVal);
		}
	}
}
