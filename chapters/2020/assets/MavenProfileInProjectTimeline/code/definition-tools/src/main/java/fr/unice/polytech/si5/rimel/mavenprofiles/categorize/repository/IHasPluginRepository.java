package fr.unice.polytech.si5.rimel.mavenprofiles.categorize.repository;

import fr.unice.polytech.si5.rimel.mavenprofiles.categorize.model.HasPlugin;
import fr.unice.polytech.si5.rimel.mavenprofiles.categorize.model.HasProperty;
import org.springframework.data.neo4j.annotation.Query;
import org.springframework.data.neo4j.repository.Neo4jRepository;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

@Repository
public interface IHasPluginRepository extends Neo4jRepository<HasPlugin, String> {

	Optional<HasPlugin> findByProfileNameAndName(String profileName, String name);

}
