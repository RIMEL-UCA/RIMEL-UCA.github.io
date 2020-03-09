package fr.unice.polytech.si5.rimel.eventprofile.commitanalyser;

import fr.unice.polytech.si5.rimel.eventprofile.domain.DiffType;
import fr.unice.polytech.si5.rimel.eventprofile.model.Event;
import fr.unice.polytech.si5.rimel.eventprofile.model.Pom;
import fr.unice.polytech.si5.rimel.eventprofile.repository.IPomRepository;
import org.apache.commons.io.FileUtils;
import org.apache.http.HttpEntity;
import org.apache.http.client.methods.CloseableHttpResponse;
import org.apache.http.client.methods.HttpGet;
import org.apache.http.impl.client.HttpClients;
import org.apache.http.util.EntityUtils;
import org.eclipse.jgit.api.DescribeCommand;
import org.eclipse.jgit.api.Git;
import org.eclipse.jgit.api.errors.GitAPIException;
import org.eclipse.jgit.lib.ObjectId;
import org.eclipse.jgit.revwalk.RevCommit;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;
import org.xmlunit.builder.DiffBuilder;
import org.xmlunit.diff.Comparison;
import org.xmlunit.diff.ComparisonType;
import org.xmlunit.diff.Diff;
import org.xmlunit.diff.Difference;

import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.nio.file.NoSuchFileException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;

@Component
public class CommitAnalyser {

	@Autowired
	private IPomRepository pomRepository;

	public void analyseCommits(final Git repo, final List<String> commits, final String pomPath, final String repositoryName, final Pom pom) throws IOException, GitAPIException {
		// For all commit that modified the pom

		for(final String commit : commits) {
			try {
				Iterator<RevCommit> jgitCommits = repo.log().all().call().iterator();
				Pom pomFromBase = this.pomRepository.findByFullRepositoryNameAndPath(pom.getFullRepositoryName(), pom.getPath()).get();
				RevCommit currentCommit;
				// Search the commit into the repo
				while (jgitCommits.hasNext()) {
					try {
						currentCommit = jgitCommits.next();
						if (currentCommit.getId().getName().equals(commit)) {
							//TODO: For the moment we only manage commits with one parent (ie merge not managed)
							final Map<ObjectId, String> map = repo
									.nameRev()
									.addPrefix("refs/heads")
									.add(ObjectId.fromString(commit))
									.call();
							final String commitMsg = currentCommit.getFullMessage();
							final String tag = repo.describe().setTarget(currentCommit).call();
							if (currentCommit.getParentCount() == 1) {
								String parentCommit = currentCommit.getParents()[0].getId().getName();
								DiffType diffType = this.analyseDiff(pomPath, repositoryName, currentCommit.getId().getName(), parentCommit);

								if (diffType != null) {
									final Event event = Event.builder().tag(tag).commitMsg(commitMsg).diffType(diffType).build();
									event.setBranches(new ArrayList<>(map.values()));
									pomFromBase.addEvent(event);
									this.pomRepository.save(pomFromBase);
								}
							} else if (currentCommit.getParentCount() == 0) {
								final Event event = Event.builder().tag(tag).commitMsg(commitMsg).build();
								event.setBranches(new ArrayList<>(map.values()));
								pomFromBase.addEvent(event);
								this.pomRepository.save(pomFromBase);
							} else if (currentCommit.getParentCount() == 2) {
								final Event event = Event.builder().tag(tag).commitMsg(commitMsg).isMerge(true).build();
								event.setBranches(new ArrayList<>(map.values()));
								pomFromBase.addEvent(event);
								this.pomRepository.save(pomFromBase);
							}
							break;
						}
					} catch (Exception e) {
						System.err.println("Exception raised : " + e.getMessage());
					}
				}
			} catch (Exception e) {
				System.err.println("Exception raised : " + e.getMessage());
			}
		}
	}

	private DiffType analyseDiff(final String pomPath, final String repositoryName, final String commit, final String parentCommit) throws IOException {
		try {
			final String newPom = this.downloadPom(pomPath, repositoryName, commit);
			final String oldPom = this.downloadPom(pomPath, repositoryName, parentCommit);
			System.out.println(newPom.length());
			System.out.println(oldPom.length());

			final Diff myDiff = DiffBuilder.compare(oldPom).withTest(newPom).ignoreComments() // [2]
					.ignoreWhitespace() // [3]
					.normalizeWhitespace().ignoreElementContentWhitespace().build();

			for(final Difference difference : myDiff.getDifferences()) {
				Comparison.Detail detail;
				if(difference.getComparison().getControlDetails().getXPath() != null) {
					detail = difference.getComparison().getControlDetails();
				} else {
					detail = difference.getComparison().getTestDetails();
				}
				if(Pattern.matches("(.*)/profiles\\[1\\]", detail.getXPath())) {
					if(difference.getComparison().getType() == ComparisonType.CHILD_NODELIST_LENGTH) {
						if((int) difference.getComparison().getControlDetails().getValue() < (int) difference.getComparison().getTestDetails().getValue()) {
							return DiffType.REMOVE;
						} else {
							return DiffType.ADD;
						}
					}
				} else if (Pattern.matches("(.*)/profile\\[[0-9]*\\]", detail.getXPath())) {
					return DiffType.MODIFICATION;
				}
			}
		} catch(Exception e){
			System.err.println(e.getMessage());
			return null;
		}
		return null;
	}

	private String downloadPom(final String pomPath, final String repositoryName, final String commit) throws IOException {
		// Download the pom
		String url = String.format("https://raw.githubusercontent.com/%s/%s/%s", repositoryName, commit, pomPath).replace(" ", "%20");
		System.out.println(url);
		HttpGet request = new HttpGet(URI.create(url));
		request.setHeader("Content-Type", "charset=utf-8");
		String fileContent = "";

		CloseableHttpResponse response = HttpClients.createDefault().execute(request);
		HttpEntity entity = response.getEntity();
		if (response.getStatusLine().getStatusCode() == 200) {
			fileContent = EntityUtils.toString(entity, StandardCharsets.UTF_8);

			// For debug purpose ONLY
            /*
            File file = new File("test.txt");
            FileUtils.writeStringToFile(file, fileContent);
            */

			response.close();

			return fileContent;
		}
		throw new NoSuchFileException("No file for " + url);
	}
}
