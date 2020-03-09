package fr.unice.polytech.si5.rimel.eventprofile.repo;

import fr.unice.polytech.si5.rimel.eventprofile.SourceDownloader;
import fr.unice.polytech.si5.rimel.eventprofile.XMLParser;
import fr.unice.polytech.si5.rimel.eventprofile.model.HasProfile;
import fr.unice.polytech.si5.rimel.eventprofile.model.MavenProfile;
import fr.unice.polytech.si5.rimel.eventprofile.model.Pom;
import fr.unice.polytech.si5.rimel.eventprofile.repository.IPomRepository;
import fr.unice.polytech.si5.rimel.eventprofile.repository.IProfileRepository;
import org.eclipse.jgit.api.Git;
import org.eclipse.jgit.api.ListBranchCommand;
import org.eclipse.jgit.lib.Ref;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;
import org.w3c.dom.Node;
import org.w3c.dom.ls.LSOutput;

import java.io.StringWriter;
import java.io.Writer;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

@Component
public class RepositoryAnalyser {

	@Autowired
	private IPomRepository pomRepository;

	@Autowired
	private IProfileRepository profileRepository;

	public Pom analyse(final Git git, final String repositoryName, final String path) throws Exception {
		System.out.println("Branches : ");
		List<Ref> branches = git.branchList().setListMode(ListBranchCommand.ListMode.ALL).call();
		for (Ref ref : branches) {
			System.out.println("Branch: " + ref.getName());
		}
		return this.downloadPom(repositoryName, path);
	}

	private Pom downloadPom(String repo, String path) throws Exception {
		String branch = "master";
		SourceDownloader sourceDownloader = new SourceDownloader(repo, branch, path);
		try {
			String content = sourceDownloader.download();
			XMLParser xmlParser = new XMLParser(content);
			List<Node> nodes = xmlParser.parseNode("profile");

			Pom pom = Pom.builder().fullRepositoryName(repo).path(path).build();
			//checkOrmUsage(content, pom);

			parsePoms(pom , nodes, xmlParser);
			return pom;
		}catch (Exception exception) {
			System.err.println(exception.getMessage());
			return null;
		}
	}

	private void parsePoms(Pom pom, List<Node> nodes, XMLParser xmlParser) throws Exception {
		List<MavenProfile> profiles = new ArrayList<>();
		Optional<Pom> pomOpt = this.pomRepository.findByFullRepositoryNameAndPath(pom.getFullRepositoryName(), pom.getPath());
		if(pomOpt.isPresent()) {
			return;
		}
		for (Node node : nodes) {
			pomOpt = this.pomRepository.findByFullRepositoryNameAndPath(pom.getFullRepositoryName(), pom.getPath());
			if(pomOpt.isPresent()) {
				pom = pomOpt.get();
			}
			LSOutput lsOutput = xmlParser.getLs().createLSOutput();
			Writer stringWriter = new StringWriter();
			lsOutput.setCharacterStream(stringWriter);
			xmlParser.getSerializer().write(node, lsOutput);
			String result = stringWriter.toString();
			XMLParser idParser = new XMLParser(result);
			//
			String profileId;
			boolean isDefault = false;
			try {
				profileId = idParser.parseNode("id").get(0).getFirstChild().getTextContent();
			} catch (Exception exception) {
				System.err.println("No ids in profile");
				continue;
			}
			Optional<MavenProfile> profileOpt = this.profileRepository.findByName(profileId);
			HasProfile hasProfile;
			MavenProfile profile;
			if(profileOpt.isPresent()) {
				profile = profileOpt.get();
				hasProfile = HasProfile.builder().mavenProfile(profile).pom(pom).build();
			} else {
				profile = MavenProfile.builder().name(profileId).build();
				hasProfile = HasProfile.builder().mavenProfile(profile).pom(pom).build();
			}
			try {
				List<Node> activation = idParser.parseNode("activeByDefault");
				//Check the activeByDefault value and if its parents is indeed an activation tag
				String parent = activation.get(0).getParentNode().getNodeName();
				if (parent.equals("activation") && activation.get(0).getTextContent().trim().equals("true")) {
					isDefault = true;
				}
			} catch (Exception exception) {
				//System.out.println("No default activation tag");
			}
			hasProfile.setDefault(isDefault);
			pom.addProfile(hasProfile);
			if (!pom.getProfiles().isEmpty()) {
				this.pomRepository.save(pom);
			}
		}
	}
}
