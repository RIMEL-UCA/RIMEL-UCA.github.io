package fr.unice.polytech.si5.rimel.mavenprofiles.categorize;

import org.apache.http.HttpEntity;
import org.apache.http.client.methods.CloseableHttpResponse;
import org.apache.http.client.methods.HttpGet;
import org.apache.http.impl.client.CloseableHttpClient;
import org.apache.http.impl.client.HttpClients;
import org.apache.http.util.EntityUtils;

import java.io.IOException;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.nio.file.NoSuchFileException;

public class SourceDownloader {
    private final CloseableHttpClient httpClient = HttpClients.createDefault();
    private String fullnameRepository;
    private String branch;
    private String pathFile;

    public SourceDownloader(String fullnameRepository, String branch, String pathFile) {
        this.fullnameRepository = fullnameRepository;
        this.branch = branch;
        this.pathFile = pathFile;
    }

    public String download() throws IOException {
        String url = String.format("https://raw.githubusercontent.com/%s/%s/%s", fullnameRepository, branch, pathFile).replace(" ", "%20");
        System.out.println(url);
        HttpGet request = new HttpGet(URI.create(url));
        request.setHeader("Content-Type", "charset=utf-8");
        String fileContent = "";

        CloseableHttpResponse response = httpClient.execute(request);
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

/*
            String json = "{ \"path\": \"demo/pom.xml\", \"repo\": { \"full_name\": \"bloomreach-forge/oai-pmh-provider\"}}";
            JsonObject resultJson = new Gson().fromJson(json, JsonObject.class);
            JsonObject repositoryJson = resultJson.get("repo").getAsJsonObject();

            String fullnameRepository = repositoryJson.get("full_name").getAsString();
            String branch = "master";
            String pathFile = resultJson.get("path").getAsString();

            System.out.println(new SourceDownloader(fullnameRepository, branch, pathFile).download());
 */