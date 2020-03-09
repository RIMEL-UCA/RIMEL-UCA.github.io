package fr.unice.polytech.si5.rimel.eventprofile.technoanalyser;

import fr.unice.polytech.si5.rimel.eventprofile.Main;
import fr.unice.polytech.si5.rimel.eventprofile.SourceDownloader;
import fr.unice.polytech.si5.rimel.eventprofile.XMLParser;
import fr.unice.polytech.si5.rimel.eventprofile.domain.Orm;
import fr.unice.polytech.si5.rimel.eventprofile.domain.StackType;
import fr.unice.polytech.si5.rimel.eventprofile.model.*;
import fr.unice.polytech.si5.rimel.eventprofile.repository.IHasTechnology;
import fr.unice.polytech.si5.rimel.eventprofile.repository.IPomRepository;
import fr.unice.polytech.si5.rimel.eventprofile.repository.ITechnologyRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.w3c.dom.Node;

import java.io.*;
import java.util.*;

@Service
public class TechnoAnalyser {
	@Autowired
	private IPomRepository pomRepository;

	@Autowired
	private ITechnologyRepository technologyRepository;

	@Autowired
	private IHasTechnology hasTechnology;

	public void analyseTechno(final String repositoryUrl, final String pomPath) throws Exception {
		// TODO: For the moment only check presence of DockerFile
		final List<String> allFiles = this.retrieveAllRepoFiles(repositoryUrl);
		// extract owner and full name of the repo
		String ownerAndFullName = repositoryUrl.split("https://github.com/")[1];
		ownerAndFullName = ownerAndFullName.replace(".git", "");
		// extract CI and Docker
		checkCIAndDockerUsage(ownerAndFullName, pomPath, allFiles);
		// analyse ORM usage in pom
		downloadPom(ownerAndFullName, pomPath);
	}

	private void checkCIAndDockerUsage(final String repositoryUrl, final String pomPath, List<String> allFiles) {
		Optional<Pom> pomOpt = this.pomRepository.findByFullRepositoryNameAndPath(repositoryUrl, pomPath);
		if (pomOpt.isPresent()) {
			Pom pom = pomOpt.get();
			// boucle docker
			System.out.println("Search for docker...");
			List<Docker> dockers = Arrays.asList(Docker.DOCKERCOMPOSE, Docker.DOCKERFILE);
			for (Docker docker : dockers) {
				System.out.println(docker);
				for (String file : allFiles) {
					if (file.contains(docker.name)) {
						// save DOCKERs
						Optional<HasTechnology> hasTechnologyOpt = this.hasTechnology.findByTechnology_Stack_AndNameAndPom_FullRepositoryName(StackType.DOCKER, docker.name, pom.getFullRepositoryName());
						if (!hasTechnologyOpt.isPresent()) {
							Optional<Technology> technologyOpt = this.technologyRepository.findByStack(StackType.DOCKER);
							if (technologyOpt.isPresent()) {
								HasTechnology hasTechnology = HasTechnology.builder().technology(technologyOpt.get()).name(docker.name).pom(pom).build();
								this.hasTechnology.save(hasTechnology);
							} else {
								HasTechnology hasTechnology = HasTechnology.builder().technology(Technology.builder().stack(StackType.DOCKER).build()).name(docker.name).pom(pom).build();
								this.hasTechnology.save(hasTechnology);
							}
						}
					}
				}
			}
			// boucle CI
			System.out.println("Search for the CI...");
			List<CI> cis = Arrays.asList(CI.CIRCLE, CI.JENKINS, CI.TRAVIS);
			for (CI ci : cis) {
				System.out.println(ci);
				for (String file : allFiles) {
					if (file.contains(ci.name)) {
						// save CIs
						Optional<HasTechnology> hasTechnologyOpt = this.hasTechnology.findByTechnology_Stack_AndNameAndPom_FullRepositoryName(StackType.CI, ci.name, pom.getFullRepositoryName());
						if (!hasTechnologyOpt.isPresent()) {
							Optional<Technology> technologyOpt = this.technologyRepository.findByStack(StackType.CI);
							if (technologyOpt.isPresent()) {
								HasTechnology hasTechnology = HasTechnology.builder().technology(technologyOpt.get()).name(ci.name).pom(pom).build();
								this.hasTechnology.save(hasTechnology);
							} else {
								HasTechnology hasTechnology = HasTechnology.builder().technology(Technology.builder().stack(StackType.CI).build()).name(ci.name).pom(pom).build();
								this.hasTechnology.save(hasTechnology);
							}
						}
					}
				}
			}

			// boucle spring
			System.out.println("Search for Spring...");
			for (String file : allFiles) {
				if (file.contains("application.properties")) {
					// save Spring
					Optional<HasTechnology> hasTechnologyOpt = this.hasTechnology.findByTechnology_StackAndPom_FullRepositoryName(StackType.SPRING, pom.getFullRepositoryName());
					if (!hasTechnologyOpt.isPresent()) {
						Optional<Technology> technologyOpt = this.technologyRepository.findByStack(StackType.SPRING);
						if (technologyOpt.isPresent()) {
							HasTechnology hasTechnology = HasTechnology.builder().technology(technologyOpt.get()).pom(pom).build();
							this.hasTechnology.save(hasTechnology);
						} else {
							HasTechnology hasTechnology = HasTechnology.builder().technology(Technology.builder().stack(StackType.SPRING).build()).pom(pom).build();
							this.hasTechnology.save(hasTechnology);
						}
					}
				}
			}
		}
	}

	private Pom downloadPom(String repo, String path) throws Exception {
		String branch = "master";
		SourceDownloader sourceDownloader = new SourceDownloader(repo, branch, path);
		try {
			String content = sourceDownloader.download();
			XMLParser xmlParser = new XMLParser(content);
			List<Node> nodes = xmlParser.parseNode("profile");

			Optional<Pom> pomOptional = pomRepository.findByFullRepositoryNameAndPath(repo, path);
			if (pomOptional.isPresent()) {
				Pom pom = pomOptional.get();
				checkOrmUsage(content, pom);
				return pom;
			}
			return null;
		} catch (Exception exception) {
			System.err.println(exception.getMessage());
			return null;
		}
	}

	private void checkOrmUsage(String pomString, Pom pom) {
		System.out.println("Search for the ORM...");
		List<Orm> varieties = Arrays.asList(Orm.ECLIPSELINK, Orm.HIBERNATE, Orm.OPENJPA, Orm.SPRINGDATA, Orm.SPRINGORM);
		for (Orm orm : varieties) {
			System.out.println(orm);
			if (pomString.contains(orm.name)) {
				// save ORMs
				Optional<HasTechnology> hasTechnologyOpt = this.hasTechnology.findByTechnology_Stack_AndNameAndPom_FullRepositoryName(StackType.ORM, orm.name, pom.getFullRepositoryName());
				if (!hasTechnologyOpt.isPresent()) {
					Optional<Technology> technologyOpt = this.technologyRepository.findByStack(StackType.ORM);
					if (technologyOpt.isPresent()) {
						HasTechnology hasTechnology = HasTechnology.builder().technology(technologyOpt.get()).name(orm.name).pom(pom).build();
						this.hasTechnology.save(hasTechnology);
					} else {
						HasTechnology hasTechnology = HasTechnology.builder().technology(Technology.builder().stack(StackType.ORM).build()).name(orm.name).pom(pom).build();
						this.hasTechnology.save(hasTechnology);
					}
				}

			}
		}
	}

	private List<String> retrieveAllRepoFiles(final String repositoryURL) throws IOException, InterruptedException {
		String[] project = repositoryURL.split("/");
		String projectName = project[project.length-1];
		Process process = Runtime.getRuntime().exec("git -C " + Main.WORKSPACE + projectName + " ls-tree -r master --name-only");
		StringBuilder output = new StringBuilder();
		BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));

		BufferedReader errorReader = new BufferedReader(new InputStreamReader(process.getErrorStream()));

		String line;
		while ((line = reader.readLine()) != null) {
			output.append(line + "\n");
		}

		int exitVal = process.waitFor();
		if (exitVal == 0) {
			System.out.println("Git get all files succeed.");
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
		return new ArrayList<>();
	}
}
