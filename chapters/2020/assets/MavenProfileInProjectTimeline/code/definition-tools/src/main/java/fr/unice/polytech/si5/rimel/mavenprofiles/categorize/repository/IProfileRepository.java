package fr.unice.polytech.si5.rimel.mavenprofiles.categorize.repository;

import fr.unice.polytech.si5.rimel.mavenprofiles.categorize.model.MavenProfile;
import fr.unice.polytech.si5.rimel.mavenprofiles.categorize.model.MavenProperty;
import org.springframework.data.neo4j.annotation.Query;
import org.springframework.data.neo4j.repository.Neo4jRepository;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

@Repository
public interface IProfileRepository extends Neo4jRepository<MavenProfile, String> {

	Optional<MavenProfile> findByName(String name);

	@Query("MATCH (n:MavenProfile)-[r:PROPERTIES]-(n2:MavenProperty) WHERE n2.name={name} AND n.name={profileName} RETURN n,r,n2")
	List<MavenProperty> matchPropertyByNameAndProfileName(String name, String profileName);
}
