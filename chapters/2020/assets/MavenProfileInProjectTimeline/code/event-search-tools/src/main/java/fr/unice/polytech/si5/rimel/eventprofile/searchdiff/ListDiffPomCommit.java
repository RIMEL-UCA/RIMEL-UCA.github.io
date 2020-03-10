package fr.unice.polytech.si5.rimel.eventprofile.searchdiff;

import fr.unice.polytech.si5.rimel.eventprofile.Main;
import org.eclipse.jgit.api.Git;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class ListDiffPomCommit {

	public List<String> listDiff(final String repositoryURL, final String pomPath) throws IOException,
			InterruptedException {

		String[] project = repositoryURL.split("/");
		String projectName = project[project.length-1];
		Process process = Runtime.getRuntime().exec("git -C " + Main.WORKSPACE + projectName + " log --all --follow -M100% --format=format:%H -- " + pomPath);
		StringBuilder output = new StringBuilder();
		BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));

		BufferedReader errorReader = new BufferedReader(new InputStreamReader(process.getErrorStream()));

		String line;
		while ((line = reader.readLine()) != null) {
			output.append(line + "\n");
		}

		int exitVal = process.waitFor();
		if (exitVal == 0) {
			System.out.println("Git search commit with pom success.");
			System.out.println(output.toString());
			return Arrays.asList(output.toString().split("\n"));
		} else {
			StringBuilder error = new StringBuilder();
			while ((line = errorReader.readLine()) != null) {
				error.append(line + " ");
			}
			System.err.println("Something went wrong in the git pom commit command.");
			System.err.println(error.toString());
			System.err.println(exitVal);
		}
		return null;
	}
}
