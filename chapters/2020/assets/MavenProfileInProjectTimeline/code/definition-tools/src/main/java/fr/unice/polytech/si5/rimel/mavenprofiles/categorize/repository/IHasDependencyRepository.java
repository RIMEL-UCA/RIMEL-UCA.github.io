package fr.unice.polytech.si5.rimel.mavenprofiles.categorize.repository;

import fr.unice.polytech.si5.rimel.mavenprofiles.categorize.model.HasDependency;
import fr.unice.polytech.si5.rimel.mavenprofiles.categorize.model.HasPlugin;
import org.springframework.data.neo4j.repository.Neo4jRepository;
import org.springframework.stereotype.Repository;

import java.util.Optional;

@Repository
public interface IHasDependencyRepository extends Neo4jRepository<HasDependency, String> {

	Optional<HasDependency> findByProfileNameAndName(String profileName, String name);

}
