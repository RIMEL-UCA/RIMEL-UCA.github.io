package fr.rimelj;

import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.stream.Collectors;
import java.util.TreeMap;

import com.bertramlabs.plugins.hcl4j.HCLParser;
import com.bertramlabs.plugins.hcl4j.HCLParserException;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;

import org.apache.commons.collections4.CollectionUtils;

import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.Response;


public final class App {
    private final ObjectMapper objectMapper = new ObjectMapper();
    private final OkHttpClient client = new OkHttpClient();
    private static final String TERRAFORM_REGISTRY_URL = "https://registry.terraform.io/v1/modules/";

    private App() {
    }

    private TerraformModuleDetails get(String id) throws IOException {
        Request request = new Request.Builder()
                .url(TERRAFORM_REGISTRY_URL + id)
                .build();

        try (Response response = client.newCall(request).execute()) {
            if (response.isSuccessful()) {
                return this.objectMapper.readValue(response.body().string(), TerraformModuleDetails.class);
            } else {
                return null;
            }

        }
    }

    private void work() {
        try {
            Map<String, List<String>> resourcesUsedByModules = new HashMap<>();
            List<TerraformModuleInfo> modulesInfo = this.objectMapper.readValue(
                    new File("modules/all-results-modules.json"), new TypeReference<List<TerraformModuleInfo>>() {
                    });
            System.out.println("Loaded " + modulesInfo.size() + " modules.");
            for (TerraformModuleInfo moduleInfo : modulesInfo) {
                var details = get(moduleInfo.id);
                if (details != null && details.root != null && details.root.resources != null) {
                    resourcesUsedByModules.put(moduleInfo.id,
                            details.root.resources.stream().map(rs -> rs.type).collect(Collectors.toList()) );
                }
                // requete vers l'api terraform

            }

            System.out.println("Saving the results");
            var resultFile = Paths.get("modules", "modules-resources-map.json").toFile();
            if (resultFile.exists()) {
                System.out.println("modules/modules-resources-map.json already exists...");
                resultFile = Paths.get("modules", "modules-resources-map" + System.currentTimeMillis() + ".json")
                        .toFile();
                System.out.println("modules/modules-resources-map.json ");
                resultFile.createNewFile();//
            }
            this.objectMapper.writeValue(resultFile, resourcesUsedByModules);
            System.out.println("Done.");

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private Map<String, Integer> findResources(String pathString) throws HCLParserException, IOException {
        File rootFile = Paths.get(pathString).toFile();
        Map<String, Integer> resourceCounts = new HashMap<>();
        if (rootFile.isDirectory()) {
            

        } else if (rootFile.isFile()) {
            
            Map results = new HCLParser().parse(rootFile, "UTF-8");

            for (Object resource : ((Map) results.get("resource")).keySet()) {
                resourceCounts.put(resource.toString(),
                        ((Map) ((Map) results.get("resource")).get(resource.toString())).size());
            }
        }

        
        return resourceCounts;

    }

    private TreeMap<Integer, String> calculateProximityScore(Map<String, Integer> resourcesCount,
            Integer similarityBonus, Integer differenceMalus)
            throws IOException {
        TreeMap<Integer, String> map = new TreeMap<>();

        Map<String, List<String>> modulesResources = this.objectMapper.readValue(
                Path.of("modules/modules-resources-map.json").toFile(), new TypeReference<Map<String, List<String>>>() {
                });
        for (Map.Entry<String, List<String>> entry : modulesResources.entrySet()) {// for every module in the terraform
                                                                                   // official registry
            String module = entry.getKey();
            List<String> resources = entry.getValue();
            int score = -differenceMalus * CollectionUtils.disjunction(resources, resourcesCount.keySet()).size()
                    + (similarityBonus * CollectionUtils.intersection(resources, resourcesCount.keySet()).size());

            map.put(score, module);

        }

        return map;

    }

    /**
     * Says hello to the world.
     * 
     * @param args The arguments of the program.
     * @throws IOException
     * @throws HCLParserException
     */
    public static void main(String[] args) throws HCLParserException, IOException {
        App app = new App();
        var files = List.of("clburlison/terraform/main.tf", "metadatamanagement/load_balancer.tf", "metadatamanagement/aws_chatbot.tf", "hotelster/terraform/main.tf");
        for (var file : files) {
            System.out.println("******************");
            TreeMap<Integer, String> treeMap = app
                    .calculateProximityScore(app.findResources(file),1,1);
            for (Entry<Integer, String> entry : treeMap.tailMap(0).entrySet()) {
                System.out.println(entry);

            }
            System.out.println("******************");

        }

        // new App().work();

    }

}
