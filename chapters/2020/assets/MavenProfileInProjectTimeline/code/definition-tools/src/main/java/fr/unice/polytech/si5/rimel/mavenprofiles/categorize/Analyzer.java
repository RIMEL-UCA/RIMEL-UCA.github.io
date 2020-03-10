package fr.unice.polytech.si5.rimel.mavenprofiles.categorize;

import fr.unice.polytech.si5.rimel.mavenprofiles.categorize.model.*;
import fr.unice.polytech.si5.rimel.mavenprofiles.categorize.repository.*;
import org.apache.commons.lang3.StringUtils;
import org.kohsuke.github.GHContent;
import org.kohsuke.github.GitHub;
import org.kohsuke.github.PagedSearchIterable;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.w3c.dom.Node;
import org.w3c.dom.ls.LSOutput;

import java.io.File;
import java.io.StringWriter;
import java.io.Writer;
import java.util.*;

@Service
public class Analyzer {
    private static final String FILENAME = "results.csv";
    private static final String TOKEN = "";
    private static final String USERNAME = "";


    @Autowired
	private IProfileRepository profileRepository;

    @Autowired
    private IHasDependencyRepository hasDependencyRepository;

	@Autowired
	private IPomRepository pomRepository;

    @Autowired
    private IHasPropertyRepository hasPropertyRepository;

    @Autowired
    private IHasPluginRepository hasPluginRepository;

    @Autowired
    private IPropertyRepository propertyRepository;

    private int pomSize = 0;

    public void run() throws Exception {
        new File(FILENAME).delete();
        doRequests(5000, 5001, 1);
    }

    private List<Pom> doRequests(int sizeBegin, int sizeEnd, int step) throws Exception {
        List<Pom> poms = new ArrayList<>();
        // Take two range to build the query Range1..Range2
        int sizeRange1 = sizeBegin;
        int sizeRange2 = sizeBegin;

        // Loop until all the interval has been requested
        while (sizeRange2 < sizeEnd) {
            sizeRange2 += step;
            this.pomSize = sizeRange2;
            String range = sizeRange1 + ".." + sizeRange2;

            System.out.println("RANGE " + range);
            // Do request
            List<Pom> getPoms = this.getAllPoms(range);
            poms.addAll(getPoms);
            // Write results for current range
            sizeRange1 = sizeRange2;
        }
        return poms;
    }

    private List<Pom> getAllPoms(final String range) throws Exception {
        List<Pom> poms = new ArrayList<>();
        GitHub github = GitHub.connect(Analyzer.USERNAME, Analyzer.TOKEN);
        PagedSearchIterable<GHContent> iterableContent = github.searchContent().q("<profile>").filename("pom.xml").language("Maven POM").size(range).list();

        System.out.println("There are " + iterableContent.getTotalCount() + " finds on this request.");
        // If there are too many result
        if (iterableContent.getTotalCount() > 1000) {
            int[] ranges = this.parseRange(range);
            // And we can reduce the step size
            if (ranges[2] > 1) {
                // We call the doRequests with a divided size
                poms.addAll(this.doRequests(ranges[0], ranges[1], ranges[2] / 2));
                return poms;
            }
        }

        int countPom = 0;
        int totalPoms = iterableContent.getTotalCount();
        // Else we unwrap github content normally
        Iterator<GHContent> iterator = iterableContent.iterator();
        while (iterator.hasNext()) {
            countPom++;
            System.out.println(String.format("========== (%s/%s) - %s%% ==========", countPom, totalPoms, (int) ((double) countPom / (double) totalPoms * 10000) / 100.0));
            GHContent content = iterator.next();
            System.out.println("Repository : " + content.getOwner().getFullName() + " Path : " + content.getPath());
            // Need to download pom
            if(content.getOwner().isFork()){
                System.err.println("THIS REPOSITORY IS A FORK.");
                continue;
            }
            Pom pom = this.downloadPom(content.getOwner().getFullName(), content.getPath());
            if (pom != null) {
                System.out.println("Downloaded.");
            }
        }
        return poms;
    }

    private int[] parseRange(String range) {
        String[] size = range.split("\\.\\.");
        return new int[]{Integer.parseInt(size[0]), Integer.parseInt(size[1]), Integer.parseInt(size[1]) - Integer.parseInt(size[0])};
    }

    private Pom downloadPom(String repo, String path) throws Exception {
        String branch = "master";
        SourceDownloader sourceDownloader = new SourceDownloader(repo, branch, path);
        try {
            String content = sourceDownloader.download();
            XMLParser xmlParser = new XMLParser(content);
            List<Node> nodes = xmlParser.parseNode("profile");

            Pom pom = Pom.builder().fullRepositoryName(repo).path(path).size(this.pomSize).build();
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
            List<MavenProperty> propertiesList = new ArrayList<>();
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
            profile = this.profileRepository.findByName(profile.getName()).get();
            System.out.println("Extracting Properties...");
            try {
                Node properties = idParser.parseNode("properties").get(0);
                for (int i = 0; i < properties.getChildNodes().getLength(); i++) {
                    if (properties.getChildNodes().item(i).getFirstChild() != null) {
                        String property = properties.getChildNodes().item(i).getFirstChild().getTextContent().trim();
                        if (StringUtils.isNotBlank(property)) {
                            String propertyName = properties.getChildNodes().item(i).getNodeName();
                            Optional<HasProperty> hasPropertyOpt = this.hasPropertyRepository.findByProfileNameAndName(profile.getName(), propertyName);
                            if (!hasPropertyOpt.isPresent()) {
                                MavenProperty prop = MavenProperty.builder().name(propertyName).build();
                                HasProperty hasProperty = HasProperty.builder().mavenProfile(profile).profileName(profile.getName()).mavenProperty(prop).name(propertyName).build();
                                profile.addProperty(hasProperty);
                                //this.profileRepository.save(profile);
                                this.hasPropertyRepository.save(hasProperty);
                            } else {
                                HasProperty hasProperty = hasPropertyOpt.get();
                                hasProperty.setWeight(hasProperty.getWeight() + 1);
                                this.hasPropertyRepository.save(hasProperty);
                            }
                        }
                    }
                }
            } catch (Exception exception) {
                System.err.println(exception.getMessage());
            }
            this.parsePlugin(idParser, profile.getName());
            this.parseDependency(idParser, profile.getName());
        }
    }

    private void parseDependency(final XMLParser idParser, final String profileName) {
        System.out.println("Extracting Dependency...");
        try {
            MavenProfile profile = this.profileRepository.findByName(profileName).get();
            List<Node> dependencies = idParser.parseNode("dependency");
            for(final Node dependency : dependencies) {
                for(int i = 0; i < dependency.getChildNodes().getLength() ; i++){
                    if(dependency.getChildNodes().item(i).getNodeName().equals("artifactId")){
                        String dependencyName = dependency.getChildNodes().item(i).getTextContent();
                        Optional<HasDependency> hasDependencyOpt = this.hasDependencyRepository.findByProfileNameAndName(profileName, dependencyName);
                        if(!hasDependencyOpt.isPresent()) {
                            MavenDependency dep = MavenDependency.builder().artifactId(dependencyName).build();
                            HasDependency hasDependency = HasDependency.builder().mavenDependency(dep).mavenProfile(profile).profileName(profileName).name(dependencyName).build();
                            profile.addDependency(hasDependency);
                            //this.profileRepository.save(profile);
                            this.hasDependencyRepository.save(hasDependency);
                        } else {
                            HasDependency hasDependency = hasDependencyOpt.get();
                            hasDependency.setWeight(hasDependency.getWeight()+1);
                            this.hasDependencyRepository.save(hasDependency);
                        }
                    }
                }
            }
        } catch (Exception exception) {
            System.err.println(exception.getMessage());
        }
    }

    private void parsePlugin(XMLParser idParser, String profileName) {
        System.out.println("Extracting Plugins...");
        try {
            MavenProfile profile = this.profileRepository.findByName(profileName).get();
            List<Node> plugins = idParser.parseNode("plugin");
            for (final Node plugin : plugins) {
                for (int i = 0; i < plugin.getChildNodes().getLength(); i++) {
                    if (plugin.getChildNodes().item(i).getNodeName().equals("artifactId")) {
                        String pluginName = plugin.getChildNodes().item(i).getTextContent();
                        Optional<HasPlugin> hasPluginOpt = this.hasPluginRepository.findByProfileNameAndName(profileName, pluginName);
                        if (!hasPluginOpt.isPresent()) {
                            MavenPlugin plug = MavenPlugin.builder().artifactId(pluginName).build();
                            HasPlugin hasPlugin = HasPlugin.builder().mavenProfile(profile).mavenPlugin(plug).profileName(profileName).name(pluginName).build();
                            profile.addPlugin(hasPlugin);
                            //this.profileRepository.save(profile);
                            this.hasPluginRepository.save(hasPlugin);
                        } else {
                            HasPlugin hasPlugin = hasPluginOpt.get();
                            hasPlugin.setWeight(hasPlugin.getWeight() + 1);
                            this.hasPluginRepository.save(hasPlugin);
                        }
                    }
                }
            }
        } catch (Exception exception) {
            System.err.println(exception.getMessage());
        }
    }

    /*
    private void writeResults(List<Pom> poms) {
        for (Pom pom : poms) {
            if(!pom.getProfiles().isEmpty()) {
                org.neo4j.driver.v1.types.Node neo4jPomNode = this.database.createPom(pom);
                for (MavenProfile profile : pom.getProfiles()) {
                    this.database.createProfile(neo4jPomNode, profile);
                }
            }
        }
        /*try (BufferedWriter writer = new BufferedWriter(new FileWriter(FILENAME, true))) {
            for (Pom pom : poms) {
                writer.write(pom.toCsv());
                writer.newLine();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }*/
}
